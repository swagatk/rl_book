'''
Solving CartPole problem using Advantage-Actor Critic (A2C) Algorithm
- Single Worker version
'''

import gymnasium as gym
from a2c import A2CAgent
import wandb 
import os
import numpy as np
import tensorflow as tf


def ac_train(env, agent, max_episodes=2000, log_freq=50, max_score=None, min_score=None, 
             stop_score=500, filename=None, wandb_log=False):
    print('Environment name: ', env.spec.id)
    print('RL Agent name:', agent.name)
    
    assert isinstance(env.action_space, gym.spaces.Discrete),\
                "A2C Agent only for discrete action spaces"

    if filename is not None:
        file = open(filename, 'w')

    if wandb_log:
        run = wandb.init(entity='swagatk', project=env.spec.id, 
            config={
                'lr_a': agent.lr_a,
                'lr_c': agent.lr_c,
                'gamma': agent.gamma,
                'agent': agent.name})

    ep_scores = []
    best_score = -np.inf
    for e in range(max_episodes):
        states, actions, rewards = [], [], []
        done = False
        state = env.reset()[0]
        ep_score = 0
        while not done:
            action = agent.policy(state)
            next_state, reward, done, _, _ = env.step(action)
            rewards.append(reward)
            states.append(state)
            actions.append(action)
            state = next_state
            ep_score += reward

            if max_score is not None and ep_score >= max_score:
                done = True
            if min_score is not None and ep_score <= min_score:
                done = True

            if done: 
                ep_scores.append(ep_score)
                # train the agent
                a_loss, c_loss = agent.train(states, actions, rewards)

        # while loop ends here
        if filename is not None:
            file.write(f'{e}\t{ep_score}\t{np.mean(ep_scores)}\t{a_loss}\t{c_loss}\n')
            file.flush()
            os.fsync(file.fileno())

        if e % log_freq == 0:
            print(f'e:{e}, ep_score:{ep_score:.2f}, avg_ep_score:{np.mean(ep_scores):.2f},\
            avg100score:{np.mean(ep_scores[-100:]):.2f}, \
                best_score:{best_score:.2f}')
        
        if wandb_log:
            wandb.log({
                'episode': e,
                'ep_score': ep_score, 
                'avg100score': np.mean(ep_scores[-100:]),
                'actor_loss': a_loss, 
                'critic_loss': c_loss,
                'mean_score': np.mean(ep_scores),
                'best_score': best_score,
            })

        if ep_score > best_score:
            best_score = ep_score
            agent.save_weights()
            print(f'Best Score: {ep_score}, episode: {e}. Model saved.')

        if np.mean(ep_scores[-100:]) > stop_score:
            print('The problem is solved in {} episodes'.format(e))
            break
    # for loop ends here
    if filename is not None:
        file.close()
    if wandb.log:
        run.finish()

if __name__ == '__main__':
    if not tf.test.is_gpu_available():
        print("No GPU found, Exiting.")
        exit(1)
    else:
        print("GPU found, proceeding with training.")
        gpus = tf.config.list_physical_devices('GPU')
        for gpu in gpus:
            print("Using GPU: ", gpu.name)

    # create the CartPole environment
    env = gym.make('CartPole-v1')
    obs_shape = env.observation_space.shape
    action_size = env.action_space.n

    print("Observation shape: ", obs_shape)
    print("Action Size: ", action_size)
    print("Max Episode steps: ", env.spec.max_episode_steps)

    # create an RL agent
    agent = A2CAgent(obs_shape, action_size)

    # train the RL agent on
    ac_train(env, agent, max_episodes=1500, log_freq=100, stop_score=500, wandb_log=True)
