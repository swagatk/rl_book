"""
This learn a deterministic policy $\pi(s)$. 
"""
import gymnasium as gym
import numpy as np

class PolicyIterationAgent():
  def __init__(self, env, gamma=0.99, max_iterations=10000, threshold=1e-10):
    self.env = env
    self.num_states = self.env.observation_space.n
    self.num_actions = self.env.action_space.n
    self.gamma = gamma
    self.max_iterations = max_iterations
    self.threshold = threshold

  def evaluate_policy(self, policy):
    value_table = np.zeros(self.num_states)
    i = 0
    while True:
      i += 1
      updated_value_table = np.copy(value_table)
      for state in range(self.num_states):
        action = policy[state]
        value_table[state] = np.sum(
            [trans_prob * (reward + self.gamma * updated_value_table[next_state])\
                for trans_prob, next_state, reward, _ in env.P[state][action]]
            )
      if (np.sum(np.fabs(updated_value_table - value_table)) < self.threshold):
        print(f'Value converges in {i} iterations.')
        break
    return value_table

  def improve_policy(self, value_table):
    policy = np.zeros(self.num_states)
    for state in range(self.num_states):
      Q_table = np.zeros(self.num_actions)
      for action in range(self.num_actions):
        Q_table[action] = np.sum(
          [trans_prob * (reward + self.gamma * value_table[next_state]) \
            for trans_prob, next_state, reward, _ in self.env.P[state][action]]
          )
      policy[state] = np.argmax(Q_table)
    return policy

  def policy_iteration(self):
    current_policy = np.zeros(self.num_states)  # action for each state
    for i in range(self.max_iterations):
      new_value_function = self.evaluate_policy(current_policy)
      new_policy = self.improve_policy(new_value_function)
      if (np.all(current_policy == new_policy)):
        print('Policy iteration converged at step %d.' %(i+1))
        current_policy = new_policy
        break
      current_policy = new_policy
    return new_policy

  def train_and_validate(self, max_episodes=10):
    # compute optimal policy
    optimal_policy = self.policy_iteration()
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

  def __del__(self):    # destructor
    self.env.close()

if __name__ == '__main__':
  #env = gym.make('FrozenLake-v1', map_name="4x4", is_slippery=True, render_mode="rgb_array")
  env = gym.make('Taxi-v3')
  state = env.reset()
  print('state:', state)
  print("Observation space dimension: ", env.observation_space.n)
  print("Action space dimension:", env.action_space.n)
  agent = PolicyIterationAgent(env)
  mean_rewards, mean_steps = agent.train_and_validate()
  print(f'mean episodic reward: {mean_rewards}, average steps per episode: {mean_steps}')