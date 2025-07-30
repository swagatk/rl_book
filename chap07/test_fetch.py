'''
Testing Gymnasium FetchEnv
This test suite is designed to validate the functionality of the FetchEnv class in the gymnasium library.

You need to install the following dependencies:
sudo apt-get install libegl1
export MUJOCO_GL=egl


'''
import gymnasium as gym
import gymnasium_robotics
import imageio

gym.register_envs(gymnasium_robotics)

env = gym.make("FetchPickAndPlace-v3", render_mode="rgb_array" )#, max_episode_steps=100)
observation, info = env.reset(seed=42)
print('Environment:', env.spec.name)
print("Observation space:", env.observation_space)
print("Action space:", env.action_space)
print("max episode steps:", env.spec.max_episode_steps)
print("Observation shape:", env.observation_space['observation'].shape)
print("Action shape:", env.action_space.shape)



frames = []
for _ in range(100):
    action = env.action_space.sample()  # Sample a random action 
    observation, reward, terminated, truncated, info = env.step(action)
    frames.append(env.render())

    if terminated or truncated:
        observation, info = env.reset()
# Save the frames as a GIF
gif_file = "fetch_env_test.gif"
imageio.mimwrite(gif_file, frames, duration=1000/60)
print(f"GIF saved as {gif_file}")
env.close()


