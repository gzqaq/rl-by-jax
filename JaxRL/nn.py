from .jax_utils import extend_and_repeat, next_rng, JaxRNG

import jax
import jax.nn.initializers as initializers
import jax.numpy as jnp
import distrax
import flax.linen as nn
from functools import partial

#####################
# Q function utils
#####################


def update_target_network(params, target_params, tau):
  return jax.tree_map(lambda q, t: tau * q + (1 - tau) * t, params,
                      target_params)


def multi_action_q_func(q_func_forward):
  """ Forward Q function with multiple actions on each state. Used as a decorator. """
  def wrapped(self, observations, actions, **kwargs):
    multi_actions = False
    bs = observations.shape[0]

    if actions.ndim == 3 and observations.ndim == 2:
      multi_actions = True

      observations = extend_and_repeat(observations, 1,
                                       actions.shape[1]).reshape(
                                           -1, observations.shape[-1])
      actions = actions.reshape(-1, actions.shape[-1])

    q_vals = q_func_forward(self, observations, actions, **kwargs)

    if multi_actions:
      q_vals = q_vals.reshape(bs, -1)

    return q_vals

  return wrapped


#####################
# Basic networds
#####################


class Scalar(nn.Module):
  init_val: float

  def setup(self):
    self.value = self.param("value", lambda _: self.init_val)

  def __call__(self):
    return self.value


class MLP(nn.Module):
  output_dim: int
  arch: str = "256-256"
  orthogonal_init: bool = False

  @nn.compact
  def __call__(self, inp):
    hidden_dims = [int(dim) for dim in self.arch.split("-")]

    x = inp
    if self.orthogonal_init:
      for n in hidden_dims:
        x = nn.Dense(
            n,
            kernel_init=initializers.orthogonal(1e-2),
            bias_init=initializers.zeros,
        )(x)
      output = nn.Dense(
          self.output_dim,
          kernel_init=initializers.orthogonal(1e-2),
          bias_init=initializers.zeros,
      )(x)
    else:
      for n in hidden_dims:
        x = nn.Dense(n)(x)
      output = nn.Dense(
          self.output_dim,
          kernel_init=initializers.variance_scaling(1e-2, "fan_in", "uniform"),
          bias_init=initializers.zeros,
      )(x)

    return output


#####################
# Q function
#####################


class QFunc(nn.Module):
  obs_dim: int
  action_dim: int
  arch: str = "256-256"
  orthogonal_init: bool = False

  @nn.compact
  @multi_action_q_func
  def __call__(self, observations, actions):
    inp = jnp.concatenate([observations, actions], axis=-1)
    output = MLP(1, self.arch, self.orthogonal_init)(inp)

    return jnp.squeeze(output, -1)

  @nn.nowrap
  def rng_keys(self):
    return ("params",)


#####################
# Policy network
#####################


class TanhGaussianPolicy(nn.Module):
  obs_dim: int
  action_dim: int
  arch: str = "256-256"
  orthogonal_init: bool = False
  log_std_multiplier: float = 1.0
  log_std_offset: float = -1.0

  def setup(self):
    self.base_network = MLP(2 * self.action_dim, self.arch,
                            self.orthogonal_init)
    self.log_std_multiplier_module = Scalar(self.log_std_multiplier)
    self.log_std_offset_module = Scalar(self.log_std_offset)

  def log_prob(self, observations, actions):
    if actions.ndim == 3:
      observations = extend_and_repeat(observations, 1, actions.shape[1])

    base_output = self.base_network(observations)
    mean, log_std = jnp.split(base_output, 2, axis=-1)

    log_std = (self.log_std_multiplier_module() * log_std +
               self.log_std_offset_module())
    log_std = jnp.clip(log_std, -20.0, 2.0)

    action_dist = distrax.Transformed(
        distrax.MultivariateNormalDiag(mean, jnp.exp(log_std)),
        distrax.Block(distrax.Tanh(), ndims=1),
    )

    return action_dist.log_prob(actions)

  def __call__(self, observations, deterministic=False, repeat=None):
    if repeat:
      observations = extend_and_repeat(observations, 1, repeat)

    base_output = self.base_network(observations)
    mean, log_std = jnp.split(base_output, 2, axis=-1)

    log_std = (self.log_std_multiplier_module() * log_std +
               self.log_std_offset_module())
    log_std = jnp.clip(log_std, -20.0, 2.0)

    action_dist = distrax.Transformed(
        distrax.MultivariateNormalDiag(mean, jnp.exp(log_std)),
        distrax.Block(distrax.Tanh(), ndims=1),
    )

    if deterministic:
      samples = jnp.tanh(mean)
      log_prob = action_dist.log_prob(samples)
    else:
      samples, log_prob = action_dist.sample_and_log_prob(
          seed=self.make_rng("noise"))

    return samples, log_prob

  @nn.nowrap
  def rng_keys(self):
    return ("params", "noise")


class SamplePolicy(object):
  def __init__(self, policy_net, net_params):
    self.policy = policy_net
    self.params = net_params

  def update_params(self, new_params):
    self.params = new_params
    return self

  @partial(jax.jit, static_argnames=("self", "deterministic"))
  def act(self, params, rng, observations, deterministic):
    return self.policy.apply(
        params,
        observations,
        deterministic,
        repeat=None,
        rngs=JaxRNG(rng)(self.policy.rng_keys()),
    )

  def __call__(self, observations, deterministic=False):
    actions, _ = self.act(self.params,
                          next_rng(),
                          observations,
                          deterministic=deterministic)

    assert jnp.all(jnp.isfinite(actions))

    return jax.device_get(actions)
