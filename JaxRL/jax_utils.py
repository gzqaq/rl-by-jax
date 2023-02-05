import jax
import jax.numpy as jnp

#####################
# JAX random
#####################


class JaxRNG(object):
  @classmethod
  def from_seed(cls, seed):
    return cls(jax.random.PRNGKey(seed))

  def __init__(self, rng) -> None:
    self.rng = rng

  def __call__(self, keys=None):
    if not keys:
      self.rng, new_rng = jax.random.split(self.rng)
      return new_rng
    elif isinstance(keys, int):
      split_rngs = jax.random.split(self.rng, num=keys + 1)
      self.rng = split_rngs[0]
      return tuple(split_rngs[1:])
    else:
      split_rngs = jax.random.split(self.rng, num=len(keys) + 1)
      self.rng = split_rngs[0]
      return {key: val for key, val in zip(keys, split_rngs[1:])}


def wrap_with_rng(rng):
  def wrapper(func):
    def wrapped(*args, **kwargs):
      nonlocal rng
      rng, new_rng = jax.random.split(rng)
      return func(new_rng, *args, **kwargs)

    return wrapped

  return wrapper


def init_rng(seed: int):
  global jax_utils_rng
  jax_utils_rng = JaxRNG.from_seed(seed)


def next_rng(*args, **kwargs):
  global jax_utils_rng
  return jax_utils_rng(*args, **kwargs)


#####################
# Array utils
#####################


def extend_and_repeat(tensor, axis: int, repeat):
  return jnp.repeat(jnp.expand_dims(tensor, axis), repeat, axis=axis)


@jax.jit
def batch_to_jax(batch):
  return jax.tree_map(jax.device_put, batch)


#####################
# Train utils
#####################


def mse_loss(pred, target):
  return jnp.mean(jnp.square(pred - target))


def value_and_multi_grad(func, n_outputs, argnums=0, has_aux=False):
  def select_output(ind):
    def wrapped(*args, **kwargs):
      if has_aux:
        x, *aux = func(*args, **kwargs)
        return (x[ind], *aux)
      else:
        x = func(*args, **kwargs)
        return x[ind]

    return wrapped

  grad_fns = tuple(
      jax.value_and_grad(select_output(i), argnums=argnums, has_aux=has_aux)
      for i in range(n_outputs))

  def multi_grad_fn(*args, **kwargs):
    grads = []
    values = []

    for grad_fn in grad_fns:
      (val, *aux), grad = grad_fn(*args, **kwargs)
      values.append(val)
      grads.append(grad)

    return (tuple(values), *aux), tuple(grads)

  return multi_grad_fn


#####################
# Evaluation utils
#####################


def collect_jax_metrics(metrics, names, prefix=None):
  collected = dict()

  for name in names:
    if name in metrics:
      collected[name] = jnp.mean(metrics[name])

  if prefix:
    collected = {f"{prefix}/{key}": val for key, val in collected.items()}

  return collected
