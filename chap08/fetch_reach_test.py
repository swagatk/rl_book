"""
This script tests the FetchReach-v3 environment from the gymnasium-robotics package.
In order to generate animation in validate() function, you need to install the following dependencies:
sudo apt update
sudo apt-get install libegl1
export MUJOCO_GL=egl
"""
import gymnasium as gym
import gymnasium_robotics
gym.register_envs(gymnasium_robotics)
import numpy as np
from PIL import Image, ImageDraw

SAVE_IMAGES = False

env = gym.make('FetchReach-v3', max_episode_steps=100, render_mode='rgb_array')
print("Observation space:", env.observation_space['observation'].shape)
print("Action space:", env.action_space.shape)
print("Action upper bound:", env.action_space.high)
print("Action lower bound:", env.action_space.low)
print("Type of action space:", type(env.action_space))
print("Observation space", env.observation_space)

max_reward = 0
min_reward = -np.inf
for i in range(3):
    obs, info = env.reset()
    obs = obs['observation']  # Extract observation from the info dict
    #print("Observation shape:", obs.shape)
    done = False
    ep_score = 0
    step = 0
    while not done:
        
        if SAVE_IMAGES:
            frame = env.render()
            image = Image.fromarray(frame)
            #image = image.resize((300, 300))
            image.save(f'fetch_reach_step_{step}.png')
            if step > 10:
                break
        action = env.action_space.sample()  # Random action
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        if reward > max_reward:
            max_reward = reward
        if reward < min_reward:
            min_reward = reward
        ep_score += reward
        step += 1
        #print(f"Step: {i}, Obs: {obs}, Reward: {reward}, Done: {done}")
        #save image into png file
    env.close()
    print(f"Episode {i} score: {ep_score}")

print(f"Max reward: {max_reward}, Min reward: {min_reward}")