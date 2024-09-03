import os
import gymnasium as gym
import numpy as np
from reinforce import REINFORCEAgent 
from utils import plot_datafile, train_agent_for_env


# instantiate a gym environment
env = gym.make('CartPole-v0')
obs_shape = env.observation_space.shape
action_size = env.action_space.n 

print('Observation shape: ', obs_shape)
print('Action Size: ', action_size)
print('Max Episode steps: ', env.spec.max_episode_steps)

# create an RL agent
agent = REINFORCEAgent(obs_shape, action_size)

# train the RL agent on
train_agent_for_env(env, agent, max_episodes=2000, filename='mc_cartpole.txt')

# plot
plot_datafile('mc_cartpole.txt', title='CartPole-v0')
