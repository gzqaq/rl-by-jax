import jax
import numpy as np
import optax
from jax import tree_map


@jax.jit
def step(params, loss_fn, optimizer, opt_state, xs, y_true):
  loss, grads = jax.value_and_grad(loss_fn)(params, xs, y_true)
  updates, opt_state = optimizer.update(grads, opt_state, params)
  params = optax.apply_updates(params, updates)
  return params, opt_state, loss


def to_batch(data, axis=-1):
  if isinstance(data, list):
    data = np.array(data)

  if data.ndim == 1:
    return np.expand_dims(data, axis=axis)
  elif data.ndim == 2:
    return data
  else:
    raise ValueError(
        f"Expect data with 1 or 2 dimensions, but get {data.ndim}!")


class ReplayBuffer(object):
  def __init__(self, max_size: int, data=None):
    self._max_size = max_size
    self._next_ind = 0
    self._size = 0
    self._initialized = False
    self._total_steps = 0

    if data:
      if self._max_size < data["observations"].shape[0]:
        self._max_size = data["observations"].shape[0]
      self.add_batch(data)

  def __len__(self):
    return self._size

  def _init_storage(self, obs_dim, action_dim):
    self._obs_dim = obs_dim
    self._action_dim = action_dim

    self._obs = np.zeros((self._max_size, obs_dim), dtype=np.float32)
    self._actions = np.zeros((self._max_size, action_dim), dtype=np.float32)
    self._next_obs = np.zeros((self._max_size, obs_dim), dtype=np.float32)
    self._rewards = np.zeros(self._max_size, dtype=np.float32)
    self._dones = np.zeros(self._max_size, dtype=np.float32)
    self._next_ind = 0
    self._size = 0
    self._initialized = True

  def add_sample(self, observation, action, reward, next_observation, done):
    if not self._initialized:
      self._init_storage(observation.size, action.size)

    self._obs[self._next_ind, :] = np.array(observation, dtype=np.float32)
    self._next_obs[self._next_ind, :] = np.array(next_observation,
                                                 dtype=np.float32)
    self._actions[self._next_ind, :] = np.array(action, dtype=np.float32)
    self._rewards[self._next_ind] = reward
    self._dones[self._next_ind] = float(done)

    if self._size < self._max_size:
      self._size += 1

    self._next_ind = (self._next_ind + 1) % self._max_size
    self._total_steps += 1

  def add_traj(self, observations, actions, rewards, next_observations, dones):
    for s, a, r, s_, d in zip(observations, actions, rewards, next_observations,
                              dones):
      self.add_sample(s, a, r, s_, d)

  def add_batch(self, batch):
    self.add_traj(
        batch["observations"],
        batch["actions"],
        batch["rewards"],
        batch["next_observations"],
        batch["dones"],
    )

  def select(self, indices):
    return dict(
        observations=self._obs[indices, ...],
        actions=self._actions[indices, ...],
        rewards=self._rewards[indices, ...],
        next_observations=self._next_observations[indices, ...],
        dones=self._dones[indices, ...],
    )

  def sample(self, batch_size):
    indices = np.random.randint(len(self), size=batch_size)
    return self.select(indices)

  def generator(self, batch_size, n_batches=None):
    i = 0

    while not n_batches or i < n_batches:
      yield self.sample(batch_size)
      i += 1

  @property
  def total_steps(self):
    return self._total_steps

  @property
  def data(self):
    return dict(
        observations=self._obs[:self._size, ...],
        actions=self._actions[:self._size, ...],
        rewards=self._rewards[:self._size, ...],
        next_observations=self._next_obs[:self._size, ...],
        dones=self._dones[:self._size, ...],
    )
