import gymnasium as gym
import numpy as np

class ValueIterationAgent():
  def __init__(self, env, gamma=0.99, max_iterations=10000):
    self.env = env
    self.num_states = self.env.observation_space.n
    self.num_actions = self.env.action_space.n
    self.gamma = gamma
    self.max_iterations = max_iterations

  def value_iteration(self, threshold=1e-20):
    value_table = np.zeros(self.num_states)
    for i in range(self.max_iterations):
      updated_value_table = np.copy(value_table)
      for state in range(self.num_states):
        Q_value = []  # Q(s,a)
        for action in range(self.num_actions):
          Q_value.append(np.sum(
          [trans_prob * \
          (reward + self.gamma * updated_value_table[next_state]) \
          for trans_prob, next_state, reward, _ in self.env.P[state][action]]))
        value_table[state] =  max(Q_value)
      if (np.sum(np.fabs(updated_value_table - value_table)) <= threshold):
        print("Value-iteration converged at iteration # %d" % (i+1))
        break
    return value_table

  def extract_policy(self, value_table):
    policy = np.zeros(self.num_states)
    for state in range(self.num_states):
      Q_table = np.zeros(self.num_actions)
      for action in range(self.num_actions):
        Q_table[action] = np.sum(
        [trans_prob * \
        (reward + self.gamma * value_table[next_state])\
        for trans_prob, next_state, reward, _ in self.env.P[state][action]])
      policy[state] = np.argmax(Q_table)
    return policy

  def train_and_validate(self, max_episodes=10):
    # Compute optimal policy
    optimal_value = self.value_iteration()
    optimal_policy = self.extract_policy(optimal_value)
    ep_rewards, ep_steps = [], []
    done = False
    for i in range(max_episodes):
      rewards = 0
      done = False
      state = self.env.reset()[0]
      step = 0
      while not done:
        step += 1
        action = optimal_policy[state]
        next_state, reward, done, _, _ = self.env.step(int(action))
        rewards += reward
        state = next_state
      ep_rewards.append(rewards)
      ep_steps.append(step)
    return np.mean(ep_rewards), np.mean(ep_steps)

  def __del__(self):
    self.env.close()
    
if __name__ == '__main__':
    #env = gym.make('FrozenLake-v1', map_name="4x4", is_slippery=True, render_mode="rgb_array")
    env = gym.make('Taxi-v3')
    state = env.reset()
    print('state:', state)
    print("Observation space dimension: ", env.observation_space.n)
    print("Action space dimension:", env.action_space.n)
    agent = ValueIterationAgent(env)
    mean_rewards, mean_steps = agent.train_and_validate()
    print(f'mean episodic reward: {mean_rewards}, average steps per episode: {mean_steps}')
