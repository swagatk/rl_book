"""
Solve Atari Problems using DQN/PER
- Frames are processed and stacked before use.
"""

import gymnasium as gym
import keras.optimizers
from wrappers import FrameStack
import keras 
from dqn import DQNAtariAgent
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def plot_model_performance(data_file, save_file='atari_dqn.png'):
    df = pd.read_csv(data_file, sep='\t')
    df.columns = ['episode', 'ep_score', 'avg_score', 'avg100_score']
    df.head()
    #ax = df.plot(x='episode', y='ep_score')
    #ax.set_ylim((0, 500))
    ax = df.plot(x='episode', y='avg_score', lw=2)
    df.plot(x='episode', y='avg100_score', ax=ax, lw=2)
    ax.grid()
    plt.savefig(save_file)


if __name__ == '__main__':

    # create an instance of gym environment
    env = gym.make('ALE/MsPacman-v5', obs_type="grayscale", render_mode='rgb_array')

    # Stack the frames using Wrapper
    env = FrameStack(env, num_stacked_frames=4)
    print('observation shape: ', env.observation_space.shape)

    n_actions = env.action_space.n 
    print('Action space dimension: ', n_actions)


    # use a CNN as DQN 
    obs_shape = (84, 84, 4) # it should match the shape of observation space.


    model = keras.Sequential([
        keras.layers.Conv2D(32, kernel_size=8, strides=4, padding='same',
                            activation='relu', kernel_initializer='he_uniform',
                            input_shape=obs_shape),
        keras.layers.MaxPooling2D(pool_size=(2,2)),
        keras.layers.Conv2D(64, kernel_size=2, strides=1, padding='same',
                            activation='relu', kernel_initializer='he_uniform'),
        keras.layers.MaxPooling2D(pool_size=(2,2)),
        keras.layers.Flatten(),
        keras.layers.Dense(128, activation='relu', kernel_initializer='he_uniform'),
        keras.layers.Dense(n_actions, activation='linear')
    ])
    model.compile(loss='mse', optimizer=keras.optimizers.Adam())


    # Create DQN PER Agent
    agent = DQNAtariAgent(obs_shape, n_actions, 
                        buffer_size=60000,
                        batch_size=64,  
                        model=model, per_flag=True)


    # Train the agent
    agent.train(env, max_episodes=200, train_freq=5, copy_freq=50, filename='pacman_dqn_per.txt')

    # plot model performance
    plot_model_performance('pacman_dqn_per.txt')