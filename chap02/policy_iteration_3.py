"""
In this code, one-hot encoding is used to represent the policy distribution $\pi(a|s)$.
"""
import gymnasium as gym
import numpy as np


class PolicyIterationAgent3():
  def __init__(self, env, gamma=0.99, threshold=1e-6):
    self.env = env
    self.num_states = self.env.observation_space.n
    self.num_actions = self.env.action_space.n
    self.gamma = gamma
    self.threshold = threshold
    self.env.reset()


  def evaluate_rewards_and_transitions(self):

    # Initialize R & T matrices
    # Rewards for transition from s -> s' with action a
    R = np.zeros((self.num_states, self.num_actions, self.num_states))
    T = np.zeros((self.num_states, self.num_actions, self.num_states))

    for state in range(self.num_states):
      for action in range(self.num_actions):
        for transition in self.env.P[state][action]:
          probability, next_state, reward, done = transition
          R[state, action, next_state] = reward
          T[state, action, next_state] = probability

        # Normalize T across state-action axes
        T[state, action, :] /= np.sum(T[state, action, :])
    return R, T

  def encode_policy(self, policy):
    """ One hot encoding of policy """
    encoded_policy = np.zeros((self.num_states, self.num_actions))
    encoded_policy[np.arange(self.num_states), policy] = 1     # one hot encoding - P(s,a)
    return encoded_policy

  def policy_evaluation(self, policy, R, T):
    value_fn = np.zeros(self.num_states)
    i = 0
    while True:
      i += 1
      prev_value_fn = value_fn.copy()
      # Q = T * (R + gamma * V[s'])
      Q = np.einsum('ijk,ijk -> ij', T, (R + self.gamma * prev_value_fn)) # Q(s,a)
      policy_prob = self.encode_policy(policy)  # \pi(s,a)
      value_fn = np.sum(policy_prob * Q, 1) # V(s)
      if np.max(np.abs(prev_value_fn - value_fn)) < self.threshold:
        print(f'value function converges in {i} steps.')
        break
    return value_fn


  def policy_improvement(self, value_fn, R, T):
    Q = np.einsum('ijk, ijk -> ij', T, (R + self.gamma * value_fn))
    policy = np.argmax(Q, axis=1)
    return policy


  def policy_iteration(self, max_iterations=100000):
    # Initialize with a random policy and initial value function
    policy = np.array([self.env.action_space.sample() for _ in range(self.num_states)])

    # Get transitions & Rewards
    R, T = self.evaluate_rewards_and_transitions()

    # iterate and improve policy

    for i in range(max_iterations): # policy improvement
      prev_policy = policy.copy()
      value_fn = self.policy_evaluation(policy, R, T)
      policy = self.policy_improvement(value_fn, R, T)
      if np.array_equal(prev_policy, policy):
        print(f'Policy iteration converges in {i+1} steps.')
        break
    return policy


  def train_and_validate(self, max_episodes=10):
    # compute optimal policy
    optimal_policy = self.policy_iteration()
    ep_rewards, ep_steps = [], []
    for i in range(max_episodes):
      state = self.env.reset()[0]
      rewards = 0
      done = False
      step = 0
      while not done:
        step += 1
        action = optimal_policy[state]
        next_state, reward, done, _ ,_= self.env.step(action)
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
  agent = PolicyIterationAgent3(env)
  mean_rewards, mean_steps = agent.train_and_validate()
  print(f'mean episodic reward: {mean_rewards}, average steps per episode: {mean_steps}')