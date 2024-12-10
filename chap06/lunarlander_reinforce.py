import gymnasium as gym
import matplotlib.pyplot as plt
from reinforce import REINFORCEAgent
from utils import train_agent_for_env, plot_datafile, validate

env = gym.make("LunarLander-v2", continuous=False, render_mode='rgb_array')
obs_shape = env.observation_space.shape
action_size = env.action_space.n
print('Environment max steps: ', env.spec.max_episode_steps)
print('Observation Space Shape: ', obs_shape)
print('Action size: ', action_size)

# create a RL agent
agent = REINFORCEAgent(obs_shape, action_size)

# Train for the environment
train_agent_for_env(env, agent, max_episodes=4000, filename='mc_lunarlander.txt')

# plot training performance
plot_datafile('mc_lunarlander.txt', title='LunarLander-v2')

# validate & save gif animation file
validate(env, agent, gif_file='lunarlander_mc.gif', wt_file='best_model.weights.h5')