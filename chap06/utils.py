import imageio
from PIL import Image
import PIL.ImageDraw as ImageDraw
import matplotlib.pyplot as plt 
import numpy as np
import pandas as pd
import os 
import gymnasium as gym

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


def validate(env, agent, num_episodes=10, gif_file=None, max_steps=None):
    frames, scores, steps = [], [], []
    for i in range(num_episodes):
        state = env.reset()[0]
        ep_reward = 0
        step = 0
        while True:
            step += 1
            if gif_file is not None and env.render_mode == 'rgb_array':
                frame = env.render()
                frames.append(_label_with_episode_number(frame, i, step))
            action = agent.policy(state) 
            next_state, reward, done, _, _ = env.step(action) 
            state = next_state
            ep_reward += reward
            if max_steps is not None and step > max_steps:
                done = True
            if done:
                scores.append(ep_reward)
                if gif_file is not None and env.render_mode == 'rgb_array':
                    frame = env.render()
                    frames.append(_label_with_episode_number(frame, i, step))
                break
        # while-loop ends here
        scores.append(ep_reward)
        steps.append(step)
        print(f'\repisode: {i}, reward: {ep_reward:.2f}, steps: {step}')
    # for-loop ends here
    if gif_file is not None:
        imageio.mimwrite(os.path.join('./', gif_file), frames, duration=1000/60)
    print('\nAverage episodic score: ', np.mean(scores))
    print('\nAverage episodic steps: ', np.mean(steps))



## plotting data file
import pandas as pd
def plot_datafile(filename:str, column_names:list, title:str):
    df = pd.read_csv(filename, sep='\t')
    df.head()
    df.columns = column_names
    ax = df.plot(x=column_names[0], y=column_names[1])
    df.plot(x=column_names[0], y=column_names[2], ax=ax, lw=3)
    df.plot(x=column_names[0], y=column_names[3], ax=ax, lw=3)
    ax.grid()
    ax.set_ylabel('Episodic Rewards')
    ax.set_title(title)


# training an agent for a problem environment
def train_agent_for_env(env, agent, max_episodes=1000, filename=None, max_score=200, min_score=-250):
    if filename is not None:
        file = open(filename, 'w')
    scores, avgscores, avg100scores = [], [], []
    best_score = 0
    for e in range(max_episodes):
        done = False
        score = 0
        state = env.reset()[0]
        state = np.expand_dims(state, axis=0)
        while not done:
            action = agent.choose_action(state)
            next_state, reward, done, _, _ = env.step(action)
            next_state = np.expand_dims(next_state, axis=0)
            agent.store_transitions(state, action, reward)
            score += reward
            state = next_state
            # terminate episode if episodic score is beyond limit
            if score > max_score or score < min_score: 
                done = True
        # end of while loop
        # train the agent at the end of each episode
        agent.train()
        scores.append(score)
        avgscores.append(np.mean(scores))
        avg100scores.append(np.mean(scores[-100:]))
        if score > best_score:
            best_score = score
            agent.save_weights('best_model.weights.h5')
            print(f'Best Score: {score}, episode: {e}. Model saved.')
            if score > 200:
                print(f'Problem is solved in {e} episodes.')
                break 
        
        if filename is not None:
            file.write(f'{e}\t{score}\t{np.mean(scores):.2f}\t{np.mean(scores[-100:]):.2f}\n')
            file.flush()
            os.fsync(file.fileno())
        if e % 100== 0:
            print('episode:{}, score: {:.2f}, avgscore: {:.2f}, avg100score: {:.2f}'\
                 .format(e, score, np.mean(scores), np.mean(scores[-100:])))
    # end of for-loop
    file.close()