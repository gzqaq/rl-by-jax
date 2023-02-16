import os
import numpy as np
import gym

from collections import defaultdict
from contextlib import (
    contextmanager,
    redirect_stderr,
    redirect_stdout,
)


@contextmanager
def suppress_output():
  with open(os.devnull, "w") as fnull:
    with redirect_stderr(fnull) as err, redirect_stdout(fnull) as out:
      yield (err, out)


with suppress_output():
  # d4rl prints out a variety of warnings
  import d4rl


def load_environment(name):
  if type(name) != str:
    return name

  with suppress_output():
    wrapped_env = gym.make(name)
  env = wrapped_env.unwrapped
  env.max_episode_steps = wrapped_env._max_episode_steps
  env.name = name

  return env


def get_d4rl_dataset(env):
  ds = d4rl.qlearning_dataset(env)

  return dict(
      observations=ds["observations"],
      actions=ds["actions"],
      next_observations=ds["next_observations"],
      rewards=ds["rewards"],
      dones=ds["terminals"].astype(np.float32),
  )


def get_sequence_dataset(env):
  ds = d4rl.qlearning_dataset(env)

  N = ds["observations"].shape[0]
  data = defaultdict(list)
  use_timeout = "timeouts" in ds

  episode_step = 0
  for i in range(N):
    done = ds["terminals"][i]

    if use_timeout:
      final = ds["timeouts"][i]
    else:
      final = episode_step == env.max_episode_steps - 1

    for key in ds:
      data[key].append(ds[key][i])

    if done or final:
      episode_step = 0
      episode_data = {}

      for key in data:
        episode_data[key] = np.array(data[key])

      yield episode_data
      data = defaultdict(list)

    episode_step += 1
