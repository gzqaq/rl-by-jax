from JaxRL.dqn import DQN, DQNPolicy
from JaxRL.utils import (
    ReplayBuffer,
    Timer,
    WandBLogger,
    define_flags_with_default,
    set_random_seed,
    print_flags,
    get_user_flags,
    prefix_metrics,
)

import absl
import gymnasium as gym

FLAGS_DEF = define_flags_with_default(
    env="CartPole-v1",
    max_traj_length=500,
    replay_buffer_size=1000000,
    seed=7,
    save_model=False,
    qf_arch="256-256",
    orthogonal_init=False,
    n_epochs=200,
    n_env_steps_per_epoch=500,
    n_train_step_per_epoch=100,
    eval_period=10,
    eval_n_trajs=5,
    batch_size=256,
    dqn=DQN.get_default_config(),
    logging=WandBLogger.get_default_config(),
)


def main(argv):
  FLAGS = absl.flags.FLAGS
  del argv

  variant = get_user_flags(FLAGS, FLAGS_DEF)
  wandb_logger = WandBLogger(config=FLAGS.logging, variant=variant)

  set_random_seed(FLAGS.seed)
