import jax


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
