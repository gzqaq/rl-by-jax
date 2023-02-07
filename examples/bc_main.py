from JaxRL.bc import BC
from JaxRL.nn import TanhGaussianPolicy, SamplePolicy
from JaxRL.samplers import TrajSampler
from JaxRL.jax_utils import batch_to_jax
from JaxRL.utils import (
    get_d4rl_dataset,
    subsample_batch,
    Timer,
    WandBLogger,
    define_flags_with_default,
    set_random_seed,
    get_user_flags,
    prefix_metrics,
)
from viskit.logging import logger, setup_logger

import absl
import gym
import d4rl
import numpy as np

FLAGS_DEF = define_flags_with_default(
    env="hopper-expert-v2",
    max_traj_length=1000,
    seed=7,
    save_model=False,

    reward_scale=1.,
    reward_bias=0.,
    clip_action=.999,

    policy_arch="256-256",
    orthogonal_init=False,
    policy_log_std_multiplier=1.,
    policy_log_std_offset=-1.,

    n_epochs=200,
    n_train_step_per_epoch=1000,
    eval_period=5,
    eval_n_trajs=5,
    batch_size=1024,

    bc=BC.get_default_config(),
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
  eval_sampler = TrajSampler(env, FLAGS.max_traj_length)

  dataset = get_d4rl_dataset(env)
  dataset[
      "rewards"] = dataset["rewards"] * FLAGS.reward_scale + FLAGS.reward_bias
  dataset["actions"] = np.clip(dataset["actions"], -FLAGS.clip_action,
                               FLAGS.clip_action)

  obs_dim = env.observation_space.shape[0]
  action_dim = env.action_space.shape[0]

  policy = TanhGaussianPolicy(
      obs_dim,
      action_dim,
      FLAGS.policy_arch,
      FLAGS.orthogonal_init,
      FLAGS.policy_log_std_multiplier,
      FLAGS.policy_log_std_offset,
  )
  bc = BC(FLAGS.bc, policy)
  sample_policy = SamplePolicy(policy, bc.train_params["policy"])

  viskit_metrics = dict()
  for i_epoch in range(FLAGS.n_epochs):
    metrics = {"epoch": i_epoch}

    with Timer() as train_timer:
      for _ in range(FLAGS.n_train_step_per_epoch):
        batch = batch_to_jax(subsample_batch(dataset, FLAGS.batch_size))
        metrics.update(prefix_metrics(bc.train(batch), "BC"))

    with Timer() as eval_timer:
      if i_epoch == 0 or (1 + i_epoch) % FLAGS.eval_period == 0:
        trajs = eval_sampler.sample(
            sample_policy.update_params(bc.train_params["policy"]),
            FLAGS.eval_n_trajs,
            deterministic=True,
        )

        metrics["average_return"] = np.mean(
            [np.sum(t["rewards"]) for t in trajs])
        metrics["average_traj_length"] = np.mean(
            [len(t["rewards"]) for t in trajs])
        metrics["average_normalized_return"] = np.mean([
            eval_sampler.env.get_normalized_score(np.sum(t["rewards"]))
            for t in trajs
        ])

        if FLAGS.save_model:
          save_data = {"BC": bc, "variant": variant, "epoch": i_epoch}
          wandb_logger.save_pickle(save_data, "model.pkl")

    metrics["train_time"] = train_timer()
    metrics["eval_time"] = eval_timer()
    metrics["epoch_time"] = train_timer() + eval_timer()
    wandb_logger.log(metrics)
    viskit_metrics.update(metrics)
    logger.record_dict(viskit_metrics)
    logger.dump_tabular(with_prefix=False, with_timestamp=False)

  if FLAGS.save_model:
    save_data = {"BC": bc, "variant": variant, "epoch": i_epoch}
    wandb_logger.save_pickle(save_data, "model.pkl")


if __name__ == "__main__":
  absl.app.run(main)
