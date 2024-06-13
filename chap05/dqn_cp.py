"""
We solve the CartPole problem using DQN/DDQN/PER etc.
The problem is considered solved if episodic reward reaches 200.
The DQN agent solves the problem in about 100 episodes
"""
import gymnasium as gym
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt 
from dqn import DQNAgent, DQNPERAgent
import imageio
from PIL import Image
import PIL.ImageDraw as ImageDraw


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
          train_freq=1, copy_freq=1, filename=None, wtfile_prefix=None):
    
    if filename is not None:
      file = open(filename, 'w')

    if wtfile_prefix is not None:
       wt_filename = wtfile_prefix + '_best_model.weights.h5'
    else:
       wt_filename = 'best_model.weights.h5'
    
    # choose between soft & hard target update (polyak averaging)
    if copy_freq < 10:
        tau = 0.1
    else:
        tau = 1.0

    best_score = 0
    scores = []
    avg_scores, avg100_scores = [], []
    global_step_cnt = 0
    for e in range(max_episodes):
        state = env.reset()[0]
        state = np.expand_dims(state, axis=0)
        done = False
        ep_reward = 0
        t = 0
        while not done:
            global_step_cnt += 1

            # take action using epsilon-greedy policy
            action = agent.get_action(state)

            # collect reward
            next_state, reward, done, _, _ = env.step(action)

            # reward engineering - important step
            reward = reward if not done else -100

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
                
            if t > 200: # problem is solved for this episode
                #print('Problem is solved for this episode')
                break
                
            # episode ends here
        if e > 100 and t > best_score:
            agent.save_model(wt_filename)
            best_score = t
        scores.append(t)
        avg_scores.append(np.mean(scores))
        avg100_scores.append(np.mean(scores[-100:]))
        if filename is not None:
          file.write(f'{e}\t{t}\t{np.mean(scores)}\t{np.mean(scores[-100:])}\n')
          file.flush()
          os.fsync(file.fileno())
        if e % 20 == 0:
            print(f'e:{e}, episodic reward: {t}, avg ep reward: {np.mean(scores):.2f}')
    print('\nEnd of training')
    file.close()

def validate(env, agent, wt_file, max_episodes=10, gif_file=None):
    
    # load weight file
    agent.load_model(wt_file)
    
    frames = []
    scores = []
    for i in range(max_episodes):
        state = env.reset()[0]
        state = np.expand_dims(state, axis=0)
        step = 0
        ep_reward = 0
        while not done: 
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

def plot_model_performance(data_file, save_file='cp_dqn.png'):
  df = pd.read_csv(data_file, sep='\t')
  df.columns = ['episode', 'ep_score', 'avg_score', 'avg100_score']
  df.head()
  ax = df.plot(x='episode', y='ep_score')
  ax.set_ylim((0, 500))
  ax.set_xlabel('Episodic Reward')
  df.plot(x='episode', y='avg_score', ax=ax, lw=2)
  df.plot(x='episode', y='avg100_score', ax=ax, lw=2)
  ax.grid()
  if save_file is not None:
    plt.savefig(save_file)


if __name__ == '__main__':

  # create a gym environment
  env = gym.make('CartPole-v0')
  obs_shape = env.observation_space.shape
  n_actions = env.action_space.n 


  # create DQN Agent
  # agent = DQNAgent(obs_shape, n_actions, 
  #                 buffer_size=2000,
  #                 batch_size=24)

  # create DQN Agent
  agent = DQNPERAgent(obs_shape, n_actions, 
                  buffer_size=2000,
                  batch_size=24)

  # train the agent
  train(env, agent, max_episodes=200, copy_freq=100, filename='cp_dqn.txt')

  # validate
  #validate(env, agent, max_episodes=10, wt_file='best_model.weights.h5', gif_file='cp_dqn.gif')

  # plot model performance
  #plot_model_performance('cp_dqn.txt')

