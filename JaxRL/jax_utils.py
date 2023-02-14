import jax
import jax.numpy as jnp

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
