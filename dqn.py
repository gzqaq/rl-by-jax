import haiku as hk
import jax
import numpy as np

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