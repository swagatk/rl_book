"""
In this code, the stochastic policy distribution $\pi(a|s)$ is learnt.
"""
import gymnasium as gym
import numpy as np

class PolicyIterationAgent2():
  def __init__(self, env, gamma=0.99, max_iterations=100000, threshold=1e-6):
    self.env = env
    self.num_states = self.env.observation_space.n
    self.num_actions = self.env.action_space.n
    self.gamma = gamma
    self.max_iterations = max_iterations
    self.threshold = threshold

  def policy_evaluation(self, policy):
    value_fn = np.zeros(self.num_states)
    i = 0
    while True:
      i += 1
      prev_value_fn = np.copy(value_fn)
      for state in range(self.num_states):
        outersum = 0
        for action in range(self.num_actions):
          q_value = np.sum(
              [trans_prob * \
              (reward + self.gamma * prev_value_fn[next_state]) \
              for trans_prob, next_state, reward, _ \
                                in self.env.P[state][action]])
          outersum += policy[state, action] * q_value
        value_fn[state] = outersum
      if (np.max(np.fabs(prev_value_fn - value_fn)) < self.threshold):
        print('Value convergences in %d iteration' %i)
        break
    return value_fn

  def policy_improvement(self, value_fn):
    q_value = np.zeros((self.num_states, self.num_actions))
    improved_policy = np.zeros((self.num_states, self.num_actions))
    for state in range(self.num_states):
      for action in range(self.num_actions):
        q_value[state, action] = np.sum(
            [trans_prob * \
            (reward + self.gamma * value_fn[next_state]) \
            for trans_prob, next_state, reward, _ \
                                in self.env.P[state][action]])
      best_action_indices = np.where(q_value[state,:] == np.max(q_value[state,:]))[0]
      for index in best_action_indices:
        improved_policy[state, index] = 1/np.size(best_action_indices)
    return improved_policy


  def policy_iteration(self):
    # start with uniform probability for all actions
    initial_policy = (1.0/self.num_actions) * np.ones((self.num_states, self.num_actions))  # \pi(s,a)
    for iter in range(self.max_iterations):
      if iter == 0:
        current_policy = initial_policy
      current_value = self.policy_evaluation(current_policy)
      improved_policy = self.policy_improvement(current_value)
      if np.allclose(current_policy, improved_policy, rtol=1e-10, atol=1e-15):
        print(f'Policy Iteration Algorithm converged in {iter+1} iterations.' )
        current_policy = improved_policy
        break
      current_policy = improved_policy
    return current_policy


  def train_and_validate(self, max_episodes=10):
    # compute optimal policy
    optimal_policy = self.policy_iteration()
    ep_rewards = []   # episodic rewards
    ep_steps = []   # steps per episode
    done = False
    for i in range(max_episodes):
      rewards = 0
      done = False
      state = self.env.reset()[0]
      step = 0
      while not done:
        step += 1
        action = np.argmax(optimal_policy[state, :])
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
  agent = PolicyIterationAgent2(env)
  mean_rewards, mean_steps = agent.train_and_validate()
  print(f'mean episodic reward: {mean_rewards}, average steps per episode: {mean_steps}')