"""
Monte Carlo GLIE Control Algorithm
- Epsilon-Greedy Policy
- Constant learning rate 
"""
import numpy as np
import sys
import random
from collections import defaultdict
import gymnasium as gym

class MCAgent():
  def __init__(self, env, alpha=0.0001, gamma=0.99, ep_decay=0.9999):
    self.env = env
    self.n_action = self.env.action_space.n
    self.alpha = alpha
    self.gamma = gamma
    print('Environment name: ', self.env.spec.name)

  def best_policy(self, Q):
    policy = dict((k, np.argmax(v)) for k, v in Q.items())
    return policy

  def epsilon_greedy_policy(self, state, Q, epsilon):
    if random.uniform(0, 1) < epsilon: # explore
      action = self.env.action_space.sample()
    else: # exploit
      action = np.argmax(Q[state,:])
    return action

  def generate_episode(self, Q, epsilon):
    states, actions, rewards = [], [], []
    state = self.env.reset()[0]
    while True:
      states.append(state)
      action = self.epsilon_greedy_policy(state, Q, epsilon)
      actions.append(action)
      next_state, reward, done, info, _ = env.step(action)
      rewards.append(reward)
      state = next_state
      if done:
        break
    return (states, actions, rewards)

  def update_Q(self, episode, Q):
    returns = 0
    states, actions, rewards = episode
    for t in range(len(states) - 1, -1, -1): # traverse in reverse order
      s = states[t]
      a = actions[t]
      r = rewards[t]
      returns += r * (self.gamma ** t) # discounted rewards
      if s not in states[:t]:   # if S is a first visit (last index is ignore)
        Q[s][a] += self.alpha * (returns - Q[s][a])
    return Q

  def mc_control(self, num_episodes=500000):
    Q = defaultdict(lambda: np.zeros(self.n_action))
    epsilon = 1.0
    eps_min = 0.0001
    decay = 0.9999
    for i in range(num_episodes):
      if i % 1000 == 0:
        print('\rEpisode: {}/{}.'.format(i, num_episodes), end="")
        sys.stdout.flush()

      episode = self.generate_episode(Q, epsilon)
      Q = self.update_Q(episode, Q)
      self.epsilon = max(epsilon * decay, eps_min)
      policy = self.best_policy(Q)
    return policy, Q

  def validate(self, policy=None, num_episodes=10):
    ep_rewards, ep_steps = [], []
    for i in range(num_episodes):
      state = self.env.reset()[0]
      done = False
      step = 0
      rewards = 0
      while not done:
        if policy is not None:  # optimal policy
          action = policy[state]
        else: # use random policy
          action = self.env.action_space.sample()
        next_state, reward, done, info, _ = self.env.step(action)
        rewards += reward
        step += 1

      ep_rewards.append(rewards)
      ep_steps.append(step)
    return np.mean(ep_rewards), np.mean(ep_steps)

  def __delete__(self):
    self.env.close()

if __name__ == '__main__':
    import gymnasium as gym

    # Problem Environment
    env = gym.make("Blackjack-v1")

    # Create a Monte-Carlo agent
    agent = MCAgent(env)

    # Learn optimal policy
    policy, Q = agent.mc_control(num_episodes=500000)

    # Compute value function
    V = dict((k, np.max(v)) for k, v in Q.items())

    # Test the optimal policy
    mean_ep_rewards, mean_ep_steps = agent.validate(policy)
    print('\nAvg Ep rewards: {}, Avg Ep steps: {} '.format(mean_ep_rewards, mean_ep_steps))