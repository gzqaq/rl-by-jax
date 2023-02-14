import numpy as np


def select_from_batch(batch: dict, indices):
  res = dict()

  for key in batch.keys():
    res[key] = batch[key][indices, ...]

  return res


def subsample_batch(batch: dict, size):
  indices = np.random.randint(batch["observations"].shape[0], size=size)
  return select_from_batch(batch, indices)
