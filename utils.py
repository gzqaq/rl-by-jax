import gymnasium as gym

def continuous_action_space(env):
  return isinstance(env.action_space, gym.spaces.box.Box)