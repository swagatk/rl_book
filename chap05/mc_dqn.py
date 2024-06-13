"""
We solve the Mountain Car Problem using DQN/DDQN/PER
- The problem is considered solved if the car reaches the peak of the hill (x=0.5) 
    in less than 200 steps. 
"""
import gymnasium as gym
import matplotlib.pyplot as plt 
import numpy as np
import os 
import sys 
import imageio
from PIL import Image
import PIL.ImageDraw as ImageDraw
import keras
from dqn import DQNAgent, DQNPERAgent
import pandas as pd

def _label_with_episode_number(frame, episode_num, step_num):
    im = Image.fromarray(frame)
    drawer = ImageDraw.Draw(im)
    if np.mean(im) < 128: # for dark image
        text_color = (255, 255, 255)
    else:
        text_color = (0, 0, 0)
        drawer.text((im.size[0]/20, im.size[1]/18),
              f'Episode: {episode_num+1}, Steps: {step_num+1}',
              fill=text_color)
    return im

def train(env, agent, max_episodes=300,
          train_freq=1, copy_freq=10, filename=None, wtfile_prefix=None):

    if filename is not None:
        file = open(filename, 'w')

    if wtfile_prefix is not None:
        wt_filename = wtfile_prefix + '_best_model.weights.h5'
    else:
        wt_filename = 'best_model.weights.h5'

    # choose between soft & hard target update 
    tau = 0.1 if copy_freq < 10 else 1.0
    max_steps = 200
    car_positions = []
    scores, avg_scores = [], []
    global_step_cnt = 0
    for e in range(max_episodes):
        # make observation
        state = env.reset()[0]
        state = np.expand_dims(state, axis=0)
        done = False
        ep_reward = 0
        t = 0
        max_pos = -99.0
        while not done:
            global_step_cnt += 1
            # take action using epsilon-greedy policy
            action = agent.get_action(state)

            # transition to next state
            # and collect reward from the environment
            next_state, reward, done, _, _ = env.step(action)
            next_state = np.expand_dims(next_state, axis=0) # (-1, 4)
            # reward engineering - important step
            if next_state[0][0] >= 0.5: 
                reward += 200
            else:
                reward = 5*abs(next_state[0][0] - state[0][0]) + 3*abs(state[0][1])
                
            # track maximum car position
            if next_state[0][0] > max_pos:
                max_pos = next_state[0][0]

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
                
            if done and t < max_steps:
                print('\nSuccessfully solved the problem in {} episodes. max_pos:{:.2f}, steps: {}\n'.format(e, max_pos, t))
                agent.save_model(wt_filename)
                
            if t >= max_steps:
                break
        # episode ends here
        car_positions.append(state[0][0])
        scores.append(ep_reward)
        avg_scores.append(np.mean(scores))
        if filename is not None:
            file.write(f'{e}\t{ep_reward}\t{np.mean(scores)}\t{max_pos}\t{t}\n' )
            file.flush()
            os.fsync(file.fileno()) # write to the file immediately
        #print on console
        print(f'\re:{e}, ep_reward: {ep_reward:.2f}, avg_ep_reward: {np.mean(scores):.2f}, ep_steps: {t}, max_pos: {max_pos:.2f}', end="")
        sys.stdout.flush()
    print('End of training')
    file.close()

def validate(env, agent, wt_file, max_episodes=10, gif_file=None):
    
    # load weight file
    agent.load_model(wt_file)
    
    frames = []
    scores = []
    for i in range(max_episodes):
        #ipdb.set_trace()
        state = env.reset()[0]
        state = np.expand_dims(state, axis=0)
        step = 0
        ep_reward = 0
        while step < 200: 
            step += 1
            if gif_file is not None and env.render_mode == 'rgb_array':
                frame = env.render()
                frames.append(_label_with_episode_number(frame, i, step))
            action = agent.get_action(state, epsilon=0.001)
            next_state, reward, done, _, _ = env.step(action)
            next_state = np.expand_dims(next_state, axis=0)
            state = next_state
            ep_reward += reward
            if done:
                if gif_file is not None and env.render_mode == 'rgb_array':
                    frame = env.render()
                    frames.append(_label_with_episode_number(frame, i, step))
                break
        # end of episode
        scores.append(ep_reward)
    # for-loop ends here
    if gif_file is not None and env.render_mode == 'rgb_array':
        imageio.mimwrite(os.path.join('./', gif_file), frames, duration=1000/60)
    print('\nAverage episodic score: ', np.mean(scores))

def plot_model_performance(data_filename, save_filename='mc_dqn.png'):
    # load data file
    df = pd.read_csv(data_filename, sep='\t', header=None, skiprows=[0])
    col_names = ['episode', 'ep_reward', 'avg_reward', 'car_position', 'ep_steps']
    df.columns = col_names
    # plot
    fig, axes = plt.subplots(3)
    df.plot(x='episode', y='ep_reward', ax=axes[0])
    df.plot(x='episode', y='avg_reward', ax=axes[0])
    axes[0].legend(loc='best')
    axes[0].set_ylabel('rewards')
    # plot car positions separately
    df.plot(x='episode', y='car_position', ax=axes[1])
    axes[1].set_ylabel('car positions')
    axes[1].grid()
    df.plot(x='episode', y='ep_steps', ax=axes[2])
    axes[2].set_ylabel('Steps per episode')

    # save plot as a file
    if save_filename is not None:
        plt.savefig(save_filename)

###############

if __name__ == '__main__':
    # create a gym environment
    env = gym.make('MountainCar-v0', render_mode='rgb_array')
    obs_shape = env.observation_space.shape
    action_shape = env.action_space.shape
    n_actions = env.action_space.n
    print('Observation shape: ', obs_shape)
    print('Action shape: ', action_shape)
    print('Action size: ', n_actions)
    print('Max episodic steps: ', env.spec.max_episode_steps)


    # Create a model
    model = keras.Sequential([
        keras.layers.Dense(30, input_shape=obs_shape, activation='relu'),
        keras.layers.Dense(60, activation='relu'),
        keras.layers.Dense(n_actions, activation='linear')
    ])
    model.compile(loss='mse', optimizer='adam')


    # create a DQN PER Agent
    agent = DQNPERAgent(obs_shape, n_actions,
                    buffer_size=20000,
                    batch_size=64,
                    model=model)

    # # create a DQN Agent
    # agent = DQNAgent(obs_shape, n_actions,
    #                  buffer_size=20000,
    #                  batch_size=64,
    #                  model=model)

    # train
    train(env, agent, max_episodes=200, copy_freq=50, filename='mc_dqn_per.txt') # try copy_freq=200 

    # validate
    # validate(env, agent, max_episodes=10, save_file='mc_dqn_per.gif')

    # plot model performance
    plot_model_performance('mc_dqn_per.txt')