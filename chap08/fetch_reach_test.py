import gymnasium as gym
import gymnasium_robotics
gym.register_envs(gymnasium_robotics)

env = gym.make('FetchReach-v3', max_episode_steps=100)
print("Observation space:", env.observation_space['observation'].shape)
print("Action space:", env.action_space.shape)
print("Action upper bound:", env.action_space.high)
print("Action lower bound:", env.action_space.low)
print("Type of action space:", type(env.action_space))
print("Observation space", env.observation_space)

for i in range(10):
    obs, info = env.reset()
    obs = obs['observation']  # Extract observation from the info dict
    #print("Observation shape:", obs.shape)
    done = False
    ep_score = 0
    while not done:
        action = env.action_space.sample()  # Random action
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        ep_score += reward
        #print(f"Step: {i}, Obs: {obs}, Reward: {reward}, Done: {done}")

    print(f"Episode {i} score: {ep_score}")