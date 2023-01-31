import numpy as np
from jax import tree_map


class ReplayBuffer(object):
  def __init__(self, max_size: int) -> None:
    self.buffer = dict(
        observations=[],
        actions=[],
        rewards=[],
        next_observations=[],
        terminals=[],
        timeouts=[],
    )

    self.max_size = max_size
    self.len = 0
    self._offset = 0

  def add(self, state, action, reward, next_state, terminal, timeout):
    if self.len < self.max_size:
      self.buffer["observations"].append(state)
      self.buffer["actions"].append(action)
      self.buffer["rewards"].append(reward)
      self.buffer["next_observations"].append(next_state)
      self.buffer["terminals"].append(terminal)
      self.buffer["timeouts"].append(timeout)
      self.len += 1
    else:
      self.buffer["observations"][self._offset] = state
      self.buffer["actions"][self._offset] = action
      self.buffer["rewards"][self._offset] = reward
      self.buffer["next_observations"][self._offset] = next_state
      self.buffer["terminals"][self._offset] = terminal
      self.buffer["timeouts"][self._offset] = timeout
      self._offset += 1

      if self._offset == self.max_size:
        self._offset = 0

  def __len__(self):
    return len(self.buffer["observations"])

  def sample(self, batch_size: int):
    indices = np.random.randint(0, self.len, (batch_size,))
    return tree_map(
        lambda x: np.array(x).take(indices, axis=0),
        self.buffer,
        is_leaf=lambda x: isinstance(x, list),
    )
