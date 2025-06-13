import gymnasium as gym
import matplotlib.pyplot as plt
from reinforce import REINFORCEAgent
from utils import validate
import wandb
import os
import numpy as np
from datetime import datetime
import tensorflow as tf
################################

CFG = dict(
    alpha = 0.0005,
    gamma = 0.99,
    agent = 'reinforce',
)
#
# training an agent for a problem environment
def train(env, agent, max_episodes=1000, filename=None, 
          ep_max_score=None, ep_min_score=None,
          stop_score=200, log_freq=100, wandb_log=False):

    if filename is not None:
        file = open(filename, 'w')

    if wandb_log:
        run = wandb.init(entity='swagatk', project=env.spec.name, config=CFG)  # use env.spec.id instead of env.spec.name
    scores = []
    best_score = -np.inf
    for e in range(max_episodes):
        done = False
        score = 0
        state = env.reset()[0]
        while not done:
            action = agent.choose_action(state)
            next_state, reward, done, _, _ = env.step(action)
            agent.store_transitions(state, action, reward)
            score += reward
            state = next_state
            # terminate episode if episodic score is beyond limit
            if ep_max_score is not None and score > ep_max_score:
                done = True
            if ep_min_score is not None and score < ep_min_score:
                done = True
        # end of while loop
        # train the agent at the end of each episode
        agent.train()
        scores.append(score)
        if score > best_score:
            best_score = score
            agent.save_weights('best_model.weights.h5')
        
        if filename is not None:
            file.write(f'{e}\t{score}\t{np.mean(scores):.2f}\t{np.mean(scores[-100:]):.2f}\n')
            file.flush()
            os.fsync(file.fileno())

        if e % log_freq == 0:
            print('episode:{}, score: {:.2f}, avgscore: {:.2f}, avg100score: {:.2f}'\
                 .format(e, score, np.mean(scores), np.mean(scores[-100:])))

        if wandb_log:
            wandb.log({
                'episode': e,
                'ep_score': score,
                'mean_score': np.mean(scores),
                'avg100score': np.mean(scores[-100:]),
            })
        
        if e > 100 and np.mean(scores[-100:]) > stop_score:
            print('The problem is solved in {} episodes'.format(e))
            break
    # for loop ends here
    if filename is not None:
        file.close()
    if wandb_log:
        run.finish()

#####################################
if __name__ == '__main__':

    if not tf.test.is_gpu_available():
        print('GPU is not available. Cannot run the code.')
        exit(1)

    # create a gym environment
    env = gym.make("LunarLander-v3", continuous=False, render_mode='rgb_array')
    obs_shape = env.observation_space.shape
    action_size = env.action_space.n
    print('Environment max steps: ', env.spec.max_episode_steps)
    print('Observation Space Shape: ', obs_shape)
    print('Action size: ', action_size)

    # create a RL agent
    agent = REINFORCEAgent(obs_shape, action_size,
                           alpha=CFG['alpha'], gamma=CFG['gamma'])

    # Train for the environment
    train(env, agent, max_episodes=4000, stop_score=200, wandb_log=True)
    

    # validate & save gif animation file
    #validate(env, agent, gif_file='lunarlander_mc.gif', wt_file='best_model.weights.h5')