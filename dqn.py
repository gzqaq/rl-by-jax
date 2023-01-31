from utils import ReplayBuffer, to_batch

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import optax

rng_key = jax.random.PRNGKey(0)


def _new_key():
  global rng_key
  rng_key, sub_key = jax.random.split(rng_key)
  return sub_key


def init(env, hidden_dims, dqn_type="vanilla"):
  action_dim = env.action_space.n

  if dqn_type != "dueling":

    def q_net(state):
      mlp = hk.nets.MLP(hidden_dims + [action_dim])
      return mlp(state)

  else:

    def q_net(state):
      hidden_layers = hk.nets.MLP(hidden_dims, activate_final=True)
      fc_a = hk.nets.MLP([action_dim])
      fc_v = hk.nets.MLP([1])

      hidden_output = hidden_layers(state)
      adv = fc_a(hidden_output)
      val = fc_v(hidden_output)

      return val + adv - adv.mean(axis=1).reshape(-1, 1)

  q_net_t = hk.without_apply_rng(hk.transform(q_net))

  # Initialize q_net
  dummy_state = np.array([env.reset()[0]])
  params = q_net_t.init(_new_key(), dummy_state)

  return params, q_net_t


def expl_action(env, params, q_net, state, eps):
  action_low = 0
  action_high = env.action_space.n

  if jax.random.uniform(_new_key()) < eps:
    action = jax.random.randint(_new_key(), (), action_low, action_high).item()
  else:
    if state.ndim == 1:
      state = np.array([state])
    action = q_net.apply(params, state).argmax().item()

  return action


def optimal_action(params, q_net, state):
  if state.ndim == 1:
    state = np.array([state])
  return q_net.apply(params, state).argmax().item()


def train(
    env,
    params,
    q_net,
    num_epochs,
    num_episodes_per_epoch,
    batch_size,
    minimal_size,
    max_buffer_size,
    learning_rate,
    gamma,
    epsilon,
    target_update,
    dqn_type="vanilla",
):
  buffer = ReplayBuffer(max_buffer_size)
  update_cnt = 0
  target_params = params.copy()
  optimizer = optax.adam(learning_rate=learning_rate)
  opt_state = optimizer.init(params)

  for i_epoch in range(num_epochs):
    epoch_loss = 0
    loss_cnt = 0
    average_return = 0
    for i_episode in range(num_episodes_per_epoch):
      episode_return = 0
      s, _ = env.reset()
      done = False

      while not done:
        a = expl_action(env, params, q_net, s, epsilon)
        s_, r, t, t_, _ = env.step(a)

        buffer.add(s, a, r, s_, t, t_)
        episode_return += r

        if len(buffer) > minimal_size:
          batch_data = buffer.sample(batch_size)
          if update_cnt == target_update:
            target_params = params.copy()
            update_cnt = 0

          params, opt_state, loss = update(
              optimizer,
              opt_state,
              params,
              target_params,
              q_net,
              batch_data,
              gamma,
              dqn_type,
          )
          update_cnt += 1

          epoch_loss += (loss - epoch_loss) / (loss_cnt + 1)
          loss_cnt += 1

        s = s_
        done = t or t_

      average_return += (episode_return - average_return) / (1 + i_episode)

    print(f"| Epoch {i_epoch+1} | Loss: {epoch_loss:.3f}", end=" | ")
    print(f"Average return: {average_return:.3f} |")

  return params


def update(optimizer, opt_state, params, target_params, q_net, batch_data,
           gamma, dqn_type):
  b_s = batch_data["observations"]
  b_a = to_batch(batch_data["actions"])
  b_r = to_batch(batch_data["rewards"])
  b_s_ = batch_data["next_observations"]
  b_t = batch_data["terminals"].astype(np.float32)
  b_t_ = batch_data["timeouts"].astype(np.float32)
  b_d = jnp.fmax(b_t, b_t_)

  if dqn_type == "double":
    max_actions = q_net.apply(params, b_s_).argmax(axis=1)
    max_next_q_vals = jnp.take_along_axis(q_net.apply(target_params, b_s_),
                                          max_actions,
                                          axis=1)
    td_targets = b_r + gamma * max_next_q_vals * (1 - b_d)
  else:
    max_next_q_vals = q_net.apply(target_params, b_s_).max(axis=1)
    td_targets = b_r + gamma * max_next_q_vals * (1 - b_d)

  def loss_fn(params):
    return optax.l2_loss(
        jnp.take_along_axis(q_net.apply(params, b_s), b_a, axis=1) -
        td_targets).mean()

  loss, grads = jax.value_and_grad(loss_fn)(params)
  updates, opt_state = optimizer.update(grads, opt_state, params)
  params = optax.apply_updates(params, updates)
  return params, opt_state, loss
