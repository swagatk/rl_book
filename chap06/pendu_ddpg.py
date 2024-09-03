import os
import sys
import numpy as np
import tensorflow as tf
import gymnasium as gym 
import matplotlib.pyplot as plt
from ddpg import DDPGAgent
from utils import plot_datafile, validate

def solve_problem(env, agent, max_episodes=500, train_freq=5, update_freq=20, filename=None):
    if update_freq <= 10:
        tau_a = 0.005
        tau_c = 0.005
    else:
        tau_a = 1.0
        tau_c = 1.0
    
    if filename is not None:
        file = open(filename, 'w')
        
    scores, ep_steps = [], []
    total_steps = 0
    best_score = -100000

    for e in range(max_episodes):
        state = env.reset()[0]
        ep_score = 0
        steps = 0
        done = False
        while not done:
            tf_state = tf.expand_dims(tf.convert_to_tensor(state), axis=0) 
            # take action 
            action = agent.policy(tf_state)
            # make transition and receive reward
            next_state, reward, done, _, _ = env.step(action)
            # store experience in the replay buffer
            agent.buffer.add((state, action, reward, next_state, done)) 
            ep_score += reward
            total_steps += 1
            steps += 1
            state = next_state  
            
            if total_steps % train_freq == 0:
                agent.experience_replay()     
                
            if total_steps % update_freq == 0:
                agent.update_targets(tau_a, tau_c)
                
            if steps > 200:
                done = True
        # end of while loop    
        if e > 100 and ep_score > best_score:
            best_score = ep_score
            agent.save_weights()
        scores.append(ep_score)
        ep_steps.append(steps)
        if filename is not None:
            file.write(f'{e}\t{ep_score}\t{np.mean(scores)}\t{np.mean(scores[-50:])}\t{steps}\n')
            file.flush()
            os.fsync(file.fileno())
        if e % 20 == 0:
            print(f'episode: {e}, score: {ep_score:.2f}, avg_score: {np.mean(scores):.2f}, \
            avg50score: {np.mean(scores[-50:]):.2f} steps: {steps}')
    # end of for loop
    file.close()


if __name__ == '__main__':
    env = gym.make('Pendulum-v1', g=9.81, render_mode="rgb_array")
    obs_shape = env.observation_space.shape
    action_shape = env.action_space.shape
    action_size = action_shape[0]
    action_ub = env.action_space.high
    action_lb = env.action_space.low
    print('Observation shape: ', obs_shape)
    print('Action shape: ', action_shape)
    print('Max episodic Steps: ', env.spec.max_episode_steps)
    print('Action space bounds: ', (action_ub[0], action_lb[0]))


    # create an agent
    agent = DDPGAgent(obs_shape, action_size, 
                  batch_size=128, buffer_capacity=20000,
                 action_upper_bound=2.0,
                 action_lower_bound=-2.0)

    # solve a problem
    solve_problem(env, agent, max_episodes=500, 
              train_freq=1, update_freq=1,
              filename='pendu_ddpg.txt')
    
    # plot data
    plot_datafile('pendu_ddpg.txt', column_names=['episode', 'score', 'avg_score', 'avg50score', 'steps'], title='Pendulum-DDPG')

    # validate a model
    agent.load_weights()
    validate(env, agent, num_episodes=5, gif_file='pendulum_ddpg.gif', max_steps=200)
