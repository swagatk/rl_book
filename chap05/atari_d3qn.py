"""
Uses a Dueling DDQN to solve the atari game problem
"""
import tensorflow as tf
import gymnasium as gym 
from wrappers import FrameStack
from dqn import DQNAtariAgent

# Creating Dueling DQN model with Conv Layers
def build_model(obs_shape, action_size):
    s_input = tf.keras.layers.Input(shape=obs_shape)
    x = tf.keras.layers.Conv2D(64, 2, activation='relu')(s_input)
    x = tf.keras.layers.MaxPool2D()(x)
    x = tf.keras.layers.Conv2D(128, 2, activation='relu')(x)
    x = tf.keras.layers.MaxPool2D()(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    x = tf.keras.layers.Dense(64, activation='relu')(x)
    v = tf.keras.layers.Dense(1, activation=None)(x)
    a = tf.keras.layers.Dense(action_size, activation=None)(x)
    lambda_func = lambda x: x[0] + (x[1] - tf.math.reduce_mean(x[1], axis=1, keepdims=True))
    Q = tf.keras.layers.Lambda(lambda_func, output_shape=((int(action_size),)), dtype=tf.float32)([v, a])
    #Q = v + (a - tf.math.reduce_mean(a, axis=1, keepdims=True))
    model = tf.keras.models.Model(s_input, Q, name='dddqn')
    model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam())
    return model


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

    # create a Dueling-DDQN model
    model = build_model(obs_shape, n_actions)


    # Create DQN PER Agent
    agent = DQNAtariAgent(obs_shape, n_actions, 
                      buffer_size=20000,
                      batch_size=64,  
                      model=model, per_flag=False)

    
    # Train the agent
    agent.train(env, max_episodes=300, train_freq=5, copy_freq=50, filename='pacman_d3qn.txt')