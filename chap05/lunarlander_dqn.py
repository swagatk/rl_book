'''
Implementation of DQN for LunarLander-v2 environment
'''
import gymnasium as gym 
import matplotlib.pyplot as plt
import numpy as np  
import wandb
import os
import sys
import tensorflow as tf
from dqn import DQNAgent, DQNPERAgent


CFG = dict(
    buffer_capacity = 50000,
    batch_size = 64,
    agent = 'ddqn_per',
)
def train(env, agent, max_episodes=300, train_freq=1, copy_freq=10, 
          max_steps_per_episode=None, wandb_log=False, wtfile_prefix=None, save_freq=None):
    print('Environment name:', env.spec.name)
    print('RL Agent name:', agent.name)

    if wtfile_prefix is not None:
        wt_filename = wtfile_prefix + '_model.weights.h5'
    else:
        wt_filename = '_model.weights.h5'

    if wandb_log:
        wandb.login()
        run = wandb.init(entity='swagatk', project=env.spec.name, config=CFG)

    
    # choose between soft & hard target update 
    tau = 0.1 if copy_freq < 10 else 1.0
    max_steps = max_steps_per_episode # max steps per episode 
    ep_scores, ep_steps = [], []
    global_step_cnt = 0
    for e in range(max_episodes):
        # make observation
        state = env.reset()[0]
        state = np.expand_dims(state, axis=0)
        done = False
        ep_reward, t = 0, 0
        while not done:
            global_step_cnt += 1
            # take action using epsilon-greedy policy
            action = agent.get_action(state)

            # transition to next state
            # and collect reward from the environment
            next_state, reward, done, _, _ = env.step(action)
            next_state = np.expand_dims(next_state, axis=0) # (-1, 4)
                
            # store experience in replay buffer
            agent.store_experience(state, action, reward, next_state, done)
            state = next_state
            ep_reward += reward
            t += 1


            # train
            if global_step_cnt % train_freq == 0:
                agent.experience_replay()

            # update target model
            if global_step_cnt % copy_freq == 0:
                agent.update_target_model(tau=tau)
                
                
            if max_steps is not None and t >= max_steps:
                done = True
        # episode ends here
        ep_scores.append(ep_reward)
        ep_steps.append(t)

        if save_freq is not None and global_step_cnt % save_freq == 0:
            agent.save_model(wt_filename)

        if wandb_log:
            wandb.log({
                'episode': e,
                'ep_reward': ep_reward,
                'avg_ep_reward': np.mean(ep_scores),
                'avg100score': np.mean(ep_scores[-100:]),
                'ep_steps': t,
            })
        print(f'\re:{e}, ep_reward: {ep_reward:.2f}, avg_ep_reward: {np.mean(ep_scores):.2f}, \
              avg100score: {np.mean(ep_scores[-100:]):.2f}, ep_steps: {t}', end="")
        sys.stdout.flush()
    # for loop ends here
    print('End of training')
    if wandb_log:
        run.finish()

def build_model(obs_shape, action_size):
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(256, input_shape=obs_shape, activation='relu',
                    kernel_initializer='he_uniform'),
        tf.keras.layers.Dense(128, activation='relu',
                            kernel_initializer='he_uniform'),
        tf.keras.layers.Dense(action_size, activation='linear',
                    kernel_initializer='he_uniform')
    ])
    model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(learning_rate=0.001))
    return model

if __name__ == '__main__':
    env = gym.make('LunarLander-v3', continuous=False, gravity=-10.0,
                        enable_wind=False, wind_power=15.0, turbulence_power=1.5)
    obs_shape = env.observation_space.shape
    action_size = env.action_space.n

    print("Observation shape: ", obs_shape)
    print("Action Size: ", action_size)
    print("Max Episode steps: ", env.spec.max_episode_steps)

    q_model = build_model(obs_shape, action_size)

    # create an RL agent
    agent = DQNPERAgent(obs_shape, action_size, 
                        batch_size=CFG['batch_size'],
                        buffer_size=CFG['buffer_capacity'],
                        model=q_model)


    # train the RL agent on
    train(env, agent, max_episodes=1000, train_freq=5, copy_freq=1000, 
          max_steps_per_episode=1000, wandb_log=True, save_freq=200, 
          wtfile_prefix='lunarlander_ddqnper')
    env.close()