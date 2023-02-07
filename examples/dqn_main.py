from JaxRL.dqn import DQN, DQNPolicy, Qnet
from JaxRL.samplers import StepSampler, TrajSampler
from JaxRL.jax_utils import batch_to_jax
from JaxRL.utils import (
    ReplayBuffer,
    Timer,
    WandBLogger,
    define_flags_with_default,
    set_random_seed,
    get_user_flags,
    prefix_metrics,
)
from viskit.logging import logger, setup_logger

import absl
import gymnasium as gym
import numpy as np

FLAGS_DEF = define_flags_with_default(
    env="CartPole-v1",
    max_traj_length=500,
    replay_buffer_size=1000000,
    seed=7,
    save_model=False,

    qf_arch="256-256",
    orthogonal_init=False,

    n_epochs=200,
    n_env_steps_per_epoch=100,
    n_train_step_per_epoch=1000,
    minimal_size=1000,
    eval_period=1,
    eval_n_trajs=5,
    batch_size=512,

    dqn=DQN.get_default_config(),
    logging=WandBLogger.get_default_config(),
)


def main(argv):
  FLAGS = absl.flags.FLAGS
  del argv

  variant = get_user_flags(FLAGS, FLAGS_DEF)
  wandb_logger = WandBLogger(config=FLAGS.logging, variant=variant)
  setup_logger(variant=variant,
               seed=FLAGS.seed,
               base_log_dir=FLAGS.logging.output_dir)

  set_random_seed(FLAGS.seed)

  env = gym.make(FLAGS.env).unwrapped

  train_sampler = StepSampler(env, FLAGS.max_traj_length)
  eval_sampler = TrajSampler(env, FLAGS.max_traj_length)
  rb = ReplayBuffer(FLAGS.replay_buffer_size)

  obs_dim = env.observation_space.shape[0]
  action_dim = env.action_space.n

  q_net = Qnet(obs_dim, action_dim, FLAGS.qf_arch, FLAGS.orthogonal_init)
  dqn = DQN(FLAGS.dqn, q_net)
  policy = DQNPolicy(q_net, dqn.config.epsilon)

  viskit_metrics = dict()
  for i_epoch in range(FLAGS.n_epochs):
    metrics = {}

    with Timer() as rollout_timer:
      train_sampler.sample(
          policy.update_q_net(dqn.train_params["q_net"]),
          FLAGS.minimal_size,
          deterministic=False,
          replay_buffer=rb,
      )
      metrics["env_steps"] = rb.total_steps
      metrics["epoch"] = i_epoch

    with Timer() as train_timer:
      for _ in range(FLAGS.n_train_step_per_epoch):
        batch = batch_to_jax(rb.sample(batch_size=FLAGS.batch_size))
        metrics.update(prefix_metrics(dqn.train(batch), "dqn"))

    with Timer() as eval_timer:
      if i_epoch == 0 or (1 + i_epoch) % FLAGS.eval_period == 0:
        trajs = eval_sampler.sample(
            policy.update_q_net(dqn.train_params["q_net"]),
            FLAGS.eval_n_trajs,
            deterministic=True,
        )

        metrics["average_return"] = np.mean(
            [np.sum(t["rewards"]) for t in trajs])
        metrics["average_traj_length"] = np.mean(
            [len(t["rewards"]) for t in trajs])

        if FLAGS.save_model:
          save_data = {"dqn": dqn, "variant": variant, "epoch": i_epoch}
          wandb_logger.save_pickle(save_data, "model.pkl")

    metrics["rollout_time"] = rollout_timer()
    metrics["train_time"] = train_timer()
    metrics["eval_time"] = eval_timer()
    metrics["epoch_time"] = rollout_timer() + train_timer() + eval_timer()
    wandb_logger.log(metrics)
    viskit_metrics.update(metrics)
    logger.record_dict(viskit_metrics)
    logger.dump_tabular(with_prefix=False, with_timestamp=False)

  if FLAGS.save_model:
    save_data = {"dqn": dqn, "variant": variant, "epoch": i_epoch}
    wandb_logger.save_pickle(save_data, "model.pkl")


if __name__ == "__main__":
  absl.app.run(main)
