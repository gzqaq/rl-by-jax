from .jax_utils import value_and_multi_grad, mse_loss, collect_jax_metrics
from .random import JaxRNG, next_rng, wrap_with_rng

import jax
import jax.numpy as jnp
import optax
from flax.training.train_state import TrainState
from functools import partial
from ml_collections import ConfigDict


class BC(object):
  @staticmethod
  def get_default_config(updates=None):
    config = ConfigDict()
    config.loss = "MLE"
    config.lr = 1e-3
    config.optim_type = "adam"

    if updates:
      config.update(ConfigDict(updates).copy_and_resolve_references())

    return config

  def __init__(self, config, policy):
    self.config = self.get_default_config(config)
    self.policy = policy
    self.obs_dim = policy.obs_dim
    self.action_dim = policy.action_dim

    self._train_states = {}
    optim_class = {"adam": optax.adam, "sgd": optax.sgd}[self.config.optim_type]

    policy_params = self.policy.init(next_rng(self.policy.rng_keys()),
                                     jnp.zeros((7, self.obs_dim)))
    self._train_states["policy"] = TrainState.create(params=policy_params,
                                                     tx=optim_class(
                                                         self.config.lr),
                                                     apply_fn=None)

    self._model_keys = ("policy",)
    self._total_steps = 0

  def train(self, batch):
    self._total_steps += 1
    self._train_states, metrics = self._train_one_step(self._train_states,
                                                       next_rng(), batch)
    return metrics

  @partial(jax.jit, static_argnames=("self"))
  def _train_one_step(self, train_states, rng, batch):
    rng_generator = JaxRNG(rng)

    def loss_fn(params):
      b_s = batch["observations"]
      b_a = batch["actions"]

      @wrap_with_rng(rng_generator())
      def forward_policy(rng, *args, **kwargs):
        return self.policy.apply(*args,
                                 **kwargs,
                                 rngs=JaxRNG(rng)(self.policy.rng_keys()))

      if self.config.loss == "MLE":
        log_probs = forward_policy(params["policy"],
                                   b_s,
                                   b_a,
                                   method=self.policy.log_prob)
        loss = -log_probs.mean()
      else:
        a_hat, _ = forward_policy(params["policy"], b_s)
        loss = mse_loss(a_hat, b_a)

      return (loss,), {f"{self.config.loss}_loss": loss}

    train_params = {key: train_states[key].params for key in self.model_keys}
    (_, aux_vals), grads = value_and_multi_grad(loss_fn,
                                                len(self.model_keys),
                                                has_aux=True)(train_params)

    new_train_states = {
        key: train_states[key].apply_gradients(grads=grads[i][key])
        for i, key in enumerate(self.model_keys)
    }
    metrics = collect_jax_metrics(aux_vals, [f"{self.config.loss}_loss"])

    return new_train_states, metrics

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
    return self.total_steps
