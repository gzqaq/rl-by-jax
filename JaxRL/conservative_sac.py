from .jax_utils import (
    JaxRNG,
    next_rng,
    wrap_with_rng,
    value_and_multi_grad,
    mse_loss,
    collect_jax_metrics,
)
from .nn import Scalar, update_target_network

import numpy as np
import jax
import jax.numpy as jnp
import optax
from copy import deepcopy
from flax.training.train_state import TrainState
from functools import partial
from ml_collections import ConfigDict


class ConservativeSAC(object):
  @staticmethod
  def get_default_config(updates=None):
    config = ConfigDict()
    config.discount = 0.99
    config.alpha_multiplier = 1.0
    config.use_automatic_entropy_tuning = True
    config.backup_entropy = False
    config.target_entropy = 0.0
    config.policy_lr = 3e-4
    config.qf_lr = 3e-4
    config.optim_type = "adam"
    config.soft_target_update_rate = 5e-3
    config.use_cql = True
    config.cql_n_actions = 10
    config.cql_importance_sample = True
    config.cql_lagrange = False
    config.cql_target_action_gap = 1.0
    config.cql_temp = 1.0
    config.cql_min_q_weight = 5.0
    config.cql_max_target_backup = False
    config.cql_clip_diff_min = -np.inf
    config.cql_clip_diff_max = np.inf

    if updates:
      config.update(ConfigDict(updates).copy_and_resolve_references())

    return config

  def __init__(self, config, policy, qf):
    self.config = self.get_default_config(config)
    self.policy = policy
    self.qf = qf
    self.obs_dim = policy.obs_dim
    self.action_dim = policy.action_dim

    self._train_states = {}

    optim_class = {"adam": optax.adam, "sgd": optax.sgd}[self.config.optim_type]

    policy_params = self.policy.init(next_rng(self.policy.rng_keys()),
                                     jnp.zeros((7, self.obs_dim)))
    qf1_params = self.qf.init(
        next_rng(self.qf.rng_keys()),
        jnp.zeros((7, self.obs_dim)),
        jnp.zeros((7, self.action_dim)),
    )
    qf2_params = self.qf.init(
        next_rng(self.qf.rng_keys()),
        jnp.zeros((7, self.obs_dim)),
        jnp.zeros((7, self.action_dim)),
    )

    self._train_states["policy"] = TrainState.create(params=policy_params,
                                                     tx=optim_class(
                                                         self.config.policy_lr),
                                                     apply_fn=None)
    self._train_states["qf1"] = TrainState.create(params=qf1_params,
                                                  tx=optim_class(
                                                      self.config.qf_lr),
                                                  apply_fn=None)
    self._train_states["qf2"] = TrainState.create(params=qf2_params,
                                                  tx=optim_class(
                                                      self.config.qf_lr),
                                                  apply_fn=None)

    self._target_qf_params = deepcopy({"qf1": qf1_params, "qf2": qf2_params})

    model_keys = ["policy", "qf1", "qf2"]

    if self.config.use_automatic_entropy_tuning:
      self.log_alpha = Scalar(0.0)
      self._train_states["log_alpha"] = TrainState.create(
          params=self.log_alpha.init(next_rng()),
          tx=optim_class(self.config.policy_lr),
          apply_fn=None,
      )
      model_keys.append("log_alpha")

    if self.config.cql_lagrange:
      self.log_alpha_prime = Scalar(1.0)
      self._train_states["log_alpha_prime"] = TrainState.create(
          params=self.log_alpha_prime.init(next_rng()),
          tx=optim_class(self.config.qf_lr),
          apply_fn=None,
      )
      model_keys.append("log_alpha_prime")

    self._model_keys = tuple(model_keys)
    self._total_steps = 0

  def train(self, batch, bc=False):
    self._total_steps += 1
    self._train_states, self._target_qf_params, metrics = self._train_step(
        self._train_states, self._target_qf_params, next_rng(), batch, bc)

    return metrics

  @partial(jax.jit, static_argnames=("self", "bc"))
  def _train_step(self, train_states, target_qf_params, rng, batch, bc=False):
    rng_generator = JaxRNG(rng)

    def loss_fn(train_params):
      b_s = batch["observations"]
      b_a = batch["actions"]
      b_r = batch["rewards"]
      b_s_ = batch["next_observations"]
      b_d = batch["dones"]

      loss_collection = {}

      @wrap_with_rng(rng_generator())
      def forward_policy(rng, *args, **kwargs):
        return self.policy.apply(*args,
                                 **kwargs,
                                 rngs=JaxRNG(rng)(self.policy.rng_keys()))

      @wrap_with_rng(rng_generator())
      def forward_qf(rng, *args, **kwargs):
        return self.qf.apply(*args,
                             **kwargs,
                             rngs=JaxRNG(rng)(self.qf.rng_keys()))

      b_a_, log_pi = forward_policy(train_params["policy"], b_s)

      if self.config.use_automatic_entropy_tuning:
        alpha_loss = (-self.log_alpha.apply(train_params["log_alpha"]) *
                      (log_pi + self.config.target_entropy).mean())
        loss_collection["log_alpha"] = alpha_loss
        alpha = (jnp.exp(self.log_alpha.apply(train_params["log_alpha"])) *
                 self.config.alpha_multiplier)
      else:
        alpha_loss = 0.0
        alpha = self.config.alpha_multiplier

      # Policy loss
      if bc:
        log_probs = forward_policy(train_params["policy"],
                                   b_s,
                                   b_a,
                                   method=self.policy.log_prob)
        policy_loss = (alpha * log_pi - log_probs).mean()
      else:
        q_new_actions = jnp.minimum(
            forward_qf(train_params["qf1"], b_s, b_a_),
            forward_qf(train_params["qf2"], b_s, b_a_),
        )
        policy_loss = (alpha * log_pi - q_new_actions).mean()

      loss_collection["policy"] = policy_loss

      # Q function loss
      q1_pred = forward_qf(train_params["qf1"], b_s, b_a)
      q2_pred = forward_qf(train_params["qf2"], b_s, b_a)

      if self.config.cql_max_target_backup:
        new_a_, next_log_pi = forward_policy(train_params["policy"],
                                             b_s_,
                                             repeat=self.config.cql_n_actions)
        target_q_vals = jnp.minimum(
            forward_qf(target_qf_params["qf1"], b_s_, new_a_),
            forward_qf(target_qf_params["qf2"], b_s_, new_a_),
        )
        max_target_indices = jnp.expand_dims(jnp.argmax(target_q_vals, axis=-1),
                                             axis=-1)
        target_q_vals = jnp.take_along_axis(target_q_vals,
                                            max_target_indices,
                                            axis=-1).squeeze(-1)
        next_log_pi - jnp.take_along_axis(
            next_log_pi, max_target_indices, axis=-1).squeeze(-1)
      else:
        new_a_, next_log_pi = forward_policy(train_params["policy"], b_s_)
        target_q_vals = jnp.minimum(
            forward_qf(target_qf_params["qf1"], b_s_, new_a_),
            forward_qf(target_qf_params["qf2"], b_s_, new_a_),
        )

      if self.config.backup_entropy:
        target_q_vals = target_q_vals - alpha * next_log_pi

      td_target = jax.lax.stop_gradient(b_r + (1.0 - b_d) *
                                        self.config.discount * target_q_vals)
      qf1_loss = mse_loss(q1_pred, td_target)
      qf2_loss = mse_loss(q2_pred, td_target)

      ### CQL
      if self.config.use_cql:
        bs = b_s.shape[0]
        cql_random_actions = jax.random.uniform(
            rng_generator(),
            shape=(bs, self.config.cql_n_actions, self.action_dim),
            minval=-1.0,
            maxval=1.0,
        )

        cql_curr_actions, cql_curr_log_pis = forward_policy(
            train_params["policy"], b_s, repeat=self.config.cql_n_actions)
        cql_next_actions, cql_next_log_pis = forward_policy(
            train_params["policy"], b_s_, repeat=self.config.cql_n_actions)

        cql_q1_rand = forward_qf(train_params["qf1"], b_s, cql_random_actions)
        cql_q2_rand = forward_qf(train_params["qf2"], b_s, cql_random_actions)
        cql_q1_curr_actions = forward_qf(train_params["qf1"], b_s,
                                         cql_curr_actions)
        cql_q2_curr_actions = forward_qf(train_params["qf2"], b_s,
                                         cql_curr_actions)
        cql_q1_next_actions = forward_qf(train_params["qf1"], b_s,
                                         cql_next_actions)
        cql_q2_next_actions = forward_qf(train_params["qf2"], b_s,
                                         cql_next_actions)

        cql_q1_concat = jnp.concatenate(
            [
                cql_q1_rand,
                jnp.expand_dims(q1_pred, 1),
                cql_q1_next_actions,
                cql_q1_curr_actions,
            ],
            axis=1,
        )
        cql_q2_concat = jnp.concatenate(
            [
                cql_q2_rand,
                jnp.expand_dims(q2_pred, 1),
                cql_q2_next_actions,
                cql_q2_curr_actions,
            ],
            axis=1,
        )
        cql_std_q1 = jnp.std(cql_q1_concat, axis=1)  # For metrics
        cql_std_q2 = jnp.std(cql_q2_concat, axis=1)  # For metrics

        if self.config.cql_importance_sample:
          random_density = np.log(0.5**self.action_dim)
          cql_q1_concat = jnp.concatenate(
              [
                  cql_q1_rand - random_density,
                  cql_q1_next_actions - cql_next_log_pis,
                  cql_q1_curr_actions - cql_curr_log_pis,
              ],
              axis=1,
          )
          cql_q2_concat = jnp.concatenate(
              [
                  cql_q2_rand - random_density,
                  cql_q2_next_actions - cql_next_log_pis,
                  cql_q2_curr_actions - cql_curr_log_pis,
              ],
              axis=1,
          )

        cql_qf1_ood = (jax.scipy.special.logsumexp(
            cql_q1_concat / self.config.cql_temp, axis=1) *
                       self.config.cql_temp)
        cql_qf2_ood = (jax.scipy.special.logsumexp(
            cql_q2_concat / self.config.cql_temp, axis=1) *
                       self.config.cql_temp)

        cql_qf1_diff = jnp.clip(
            cql_qf1_ood - q1_pred,
            self.config.cql_clip_diff_min,
            self.config.cql_clip_diff_max,
        ).mean()
        cql_qf2_diff = jnp.clip(
            cql_qf2_ood - q2_pred,
            self.config.cql_clip_diff_min,
            self.config.cql_clip_diff_max,
        ).mean()

        if self.config.cql_lagrange:
          alpha_prime = jnp.clip(
              jnp.exp(
                  self.log_alpha_prime.apply(train_params["log_alpha_prime"])),
              a_min=0.0,
              a_max=1e6,
          )
          cql_min_qf1_loss = (
              alpha_prime * self.config.cql_min_q_weight *
              (cql_qf1_diff - self.config.cql_target_action_gap))
          cql_min_qf2_loss = (
              alpha_prime * self.config.cql_min_q_weight *
              (cql_qf2_diff - self.config.cql_target_action_gap))

          alpha_prime_loss = (-cql_min_qf1_loss - cql_min_qf2_loss) / 2
          loss_collection["log_alpha_prime"] = alpha_prime_loss

        else:
          cql_min_qf1_loss = cql_qf1_diff * self.config.cql_min_q_weight
          cql_min_qf2_loss = cql_qf2_diff * self.config.cql_min_q_weight
          alpha_prime_loss = 0.0
          alpha_prime = 0.0

        qf1_loss = qf1_loss + cql_min_qf1_loss
        qf2_loss = qf2_loss + cql_min_qf2_loss

      loss_collection["qf1"] = qf1_loss
      loss_collection["qf2"] = qf2_loss

      del b_s, b_a, b_r, b_s_, b_d, b_a_
      return tuple(loss_collection[key] for key in self.model_keys), locals()

    train_params = {key: train_states[key].params for key in self.model_keys}
    (_, aux_vals), grads = value_and_multi_grad(loss_fn,
                                                len(self.model_keys),
                                                has_aux=True)(train_params)

    new_train_states = {
        key: train_states[key].apply_gradients(grads=grads[i][key])
        for i, key in enumerate(self.model_keys)
    }
    new_target_qf_params = {}
    new_target_qf_params["qf1"] = update_target_network(
        new_train_states["qf1"].params,
        target_qf_params["qf1"],
        self.config.soft_target_update_rate,
    )
    new_target_qf_params["qf2"] = update_target_network(
        new_train_states["qf2"].params,
        target_qf_params["qf2"],
        self.config.soft_target_update_rate,
    )

    metrics = collect_jax_metrics(
        aux_vals,
        [
            "log_pi",
            "policy_loss",
            "qf1_loss",
            "qf2_loss",
            "alpha_loss",
            "alpha",
            "q1_pred",
            "q2_pred",
            "target_q_vals",
        ],
    )
    if self.config.use_cql:
      metrics.update(
          collect_jax_metrics(
              aux_vals,
              [
                  "cql_std_q1",
                  "cql_std_q2",
                  "cql_q1_rand",
                  "cql_q2_rand"
                  "cql_qf1_diff",
                  "cql_qf2_diff",
                  "cql_min_qf1_loss",
                  "cql_min_qf2_loss",
                  "cql_q1_current_actions",
                  "cql_q2_current_actions"
                  "cql_q1_next_actions",
                  "cql_q2_next_actions",
                  "alpha_prime",
                  "alpha_prime_loss",
              ],
              "cql",
          ))

    return new_train_states, new_target_qf_params, metrics

  @property
  def model_keys(self):
    return self._model_keys

  @property
  def train_states(self):
    return self._train_states

  @property
  def train_params(self):
    return {key: self.train_states[key].params for key in self.model_keys}

  @property
  def total_steps(self):
    return self._total_steps
