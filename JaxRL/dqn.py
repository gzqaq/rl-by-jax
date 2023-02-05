from .jax_utils import (
    next_rng,
    wrap_with_rng,
    JaxRNG,
    mse_loss,
    value_and_multi_grad,
    collect_jax_metrics,
)
from .nn import MLP

import jax
import jax.numpy as jnp
import flax.linen as nn
import optax
from copy import deepcopy
from flax.training.train_state import TrainState
from functools import partial
from ml_collections import ConfigDict


class Qnet(nn.Module):
  obs_dim: int
  action_dim: int
  arch: str = "256-256"
  orthogonal_init: bool = False

  @nn.compact
  def __call__(self, observations):
    output = MLP(self.action_dim, self.arch, self.orthogonal_init)(observations)

    return output

  @nn.nowrap
  def rng_keys(self):
    return ("params",)


class DQNPolicy(object):
  q_net: Qnet
  eps: float = 0.1

  def __init__(self, q_net: Qnet, eps: float = 0.1):
    self.q_net = q_net
    self.eps = eps
    self.action_dim = self.q_net.action_dim

  def update_q_net(self, params):
    self.params = params

    return self

  @partial(jax.jit, static_argnames=("self"))
  def act(self, params, rng, observations):
    return self.q_net.apply(params,
                            observations,
                            rngs=JaxRNG(rng)(
                                self.q_net.rng_keys())).argmax(axis=-1)

  def __call__(self, observations, deterministic=False):
    if deterministic:
      actions = self.act(self.params, next_rng(), observations)
    else:
      if jax.random.uniform(next_rng()) < self.eps:
        actions = jax.random.randint(next_rng(), (1,), 0, self.action_dim)
      else:
        actions = self.act(self.params, next_rng(), observations)

    return jax.device_get(actions)


class DQN(object):
  @staticmethod
  def get_default_config(updates=None):
    config = ConfigDict()
    config.discount = 0.98
    config.epsilon = 0.01
    config.target_update = 10
    config.dqn_type = "double"
    config.lr = 2e-3

    if updates:
      config.update(ConfigDict(updates).copy_and_resolve_references())

    return config

  def __init__(self, config, q_net: Qnet):
    self.config = self.get_default_config(config)
    self.q_net = q_net
    self.obs_dim = q_net.obs_dim
    self.action_dim = q_net.action_dim

    self._train_states = {}

    q_params = self.q_net.init(next_rng(self.q_net.rng_keys()),
                               jnp.zeros((10, self.obs_dim)))
    self._train_states["q_net"] = TrainState.create(params=q_params,
                                                    tx=optax.adam(
                                                        self.config.lr),
                                                    apply_fn=None)
    self._target_qf_params = deepcopy({"q_net": q_params})

    model_keys = ["q_net"]
    self._model_keys = tuple(model_keys)
    self._total_steps = 0

  def train(self, batch):
    self._total_steps += 1
    self._train_states, self._target_qf_params, metrics = self._train_step(
        self._train_states, self._target_qf_params, next_rng(), batch)

    return metrics

  @partial(jax.jit, static_argnames="self")
  def _train_step(self, train_states, target_qf_params, rng, batch):
    rng_generator = JaxRNG(rng)

    def loss_fn(params):
      b_s = batch["observations"]
      b_a = batch["actions"].astype(int)
      b_s_ = batch["next_observations"]
      b_r = batch["rewards"]
      b_d = batch["dones"]

      @wrap_with_rng(rng_generator())
      def forward_qf(rng, *args, **kwargs):
        return self.q_net.apply(*args,
                                **kwargs,
                                rngs=JaxRNG(rng)(self.q_net.rng_keys()))

      if self.config.dqn_type == "double":
        max_actions =jnp.expand_dims(forward_qf(params["q_net"], b_s).argmax(axis=1), axis=-1)
        target_q_vals = jnp.take_along_axis(forward_qf(
            target_qf_params["q_net"], b_s_),
                                            max_actions,
                                            axis=1)
      else:
        target_q_vals = forward_qf(target_qf_params["q_net"], b_s_).max(axis=1)

      td_target = jax.lax.stop_gradient(b_r +
                                        self.config.discount * target_q_vals *
                                        (1 - b_d))
      q_vals = jnp.take_along_axis(forward_qf(params["q_net"], b_s),
                                   b_a,
                                   axis=1)

      q_loss = mse_loss(q_vals, td_target)

      return (q_loss,), locals()

    train_params = {key: train_states[key].params for key in self.model_keys}
    (_, aux_vals), grads = value_and_multi_grad(loss_fn,
                                                len(self.model_keys),
                                                has_aux=True)(train_params)

    new_train_states = {
        key: train_states[key].apply_gradients(grads=grads[i][key])
        for i, key in enumerate(self.model_keys)
    }
    if self.total_steps % self.config.target_update == 0:
      target_qf_params["q_net"] = deepcopy(new_train_states["q_net"])

    metrics = collect_jax_metrics(aux_vals,
                                  ["q_vals", "target_q_vals", "q_loss"])

    return new_train_states, target_qf_params, metrics

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
