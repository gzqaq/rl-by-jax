from .jax_utils import init_rng

import absl.flags
import cloudpickle as pickle
import numpy as np
import os
import pprint
import random
import tempfile
import time
import uuid
import wandb
from absl import logging
from copy import copy
from ml_collections import ConfigDict
from ml_collections.config_dict import config_dict
from ml_collections.config_flags import config_flags
from socket import gethostname


def define_flags_with_default(**kwargs):
  for key, val in kwargs.items():
    if isinstance(val, ConfigDict):
      config_flags.DEFINE_config_dict(key, val)
    elif isinstance(val, bool):
      absl.flags.DEFINE_bool(key, val, "automatically defined flag")
    elif isinstance(val, int):
      absl.flags.DEFINE_integer(key, val, "automatically defined flag")
    elif isinstance(val, float):
      absl.flags.DEFINE_float(key, val, "automatically defined flag")
    elif isinstance(val, str):
      absl.flags.DEFINE_string(key, val, "automatically defined flag")
    else:
      raise ValueError("Incorrect value type")

  return kwargs


def set_random_seed(seed):
  np.random.seed(seed)
  random.seed(seed)
  init_rng(seed)


def print_flags(flags, flags_def):
  logging.info("Running training with hyperparameters: \n{}".format(
      pprint.pformat([
          "{}: {}".format(key, val)
          for key, val in get_user_flags(flags, flags_def).items()
      ])))


def get_user_flags(flags, flags_def):
  output = {}
  for key in flags_def:
    val = getattr(flags, key)
    if isinstance(val, ConfigDict):
      output.update(flatten_config_dict(val, prefix=key))
    else:
      output[key] = val

  return output


def flatten_config_dict(config, prefix=None):
  output = {}
  for key, val in config.items():
    if prefix is not None:
      next_prefix = "{}.{}".format(prefix, key)
    else:
      next_prefix = key
    if isinstance(val, ConfigDict):
      output.update(flatten_config_dict(val, prefix=next_prefix))
    else:
      output[next_prefix] = val
  return output


def prefix_metrics(metrics, prefix):
  return {"{}/{}".format(prefix, key): value for key, value in metrics.items()}


class Timer(object):
  def __init__(self):
    self._time = None

  def __enter__(self):
    self._start_time = time.time()
    return self

  def __exit__(self,):
    self._time = time.time() - self._start_time

  def __call__(self):
    return self._time


class WandBLogger(object):
  @staticmethod
  def get_default_config(updates=None):
    config = ConfigDict()
    config.online = False
    config.prefix = "JaxCQL"
    config.project = "sac"
    config.output_dir = "/tmp/JaxCQL"
    config.random_delay = 0.0
    config.experiment_id = config_dict.placeholder(str)
    config.anonymous = config_dict.placeholder(str)
    config.notes = config_dict.placeholder(str)

    if updates is not None:
      config.update(ConfigDict(updates).copy_and_resolve_references())
    return config

  def __init__(self, config, variant):
    self.config = self.get_default_config(config)

    if self.config.experiment_id is None:
      self.config.experiment_id = uuid.uuid4().hex

    if self.config.prefix != "":
      self.config.project = "{}--{}".format(self.config.prefix,
                                            self.config.project)

    if self.config.output_dir == "":
      self.config.output_dir = tempfile.mkdtemp()
    else:
      self.config.output_dir = os.path.join(self.config.output_dir,
                                            self.config.experiment_id)
      os.makedirs(self.config.output_dir, exist_ok=True)

    self._variant = copy(variant)

    if "hostname" not in self._variant:
      self._variant["hostname"] = gethostname()

    if self.config.random_delay > 0:
      time.sleep(np.random.uniform(0, self.config.random_delay))

    self.run = wandb.init(
        reinit=True,
        config=self._variant,
        project=self.config.project,
        dir=self.config.output_dir,
        id=self.config.experiment_id,
        anonymous=self.config.anonymous,
        notes=self.config.notes,
        settings=wandb.Settings(
            start_method="thread",
            _disable_stats=True,
        ),
        mode="online" if self.config.online else "offline",
    )

  def log(self, *args, **kwargs):
    self.run.log(*args, **kwargs)

  def save_pickle(self, obj, filename):
    with open(os.path.join(self.config.output_dir, filename), "wb") as fout:
      pickle.dump(obj, fout)

  @property
  def experiment_id(self):
    return self.config.experiment_id

  @property
  def variant(self):
    return self.config.variant

  @property
  def output_dir(self):
    return self.config.output_dir


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
