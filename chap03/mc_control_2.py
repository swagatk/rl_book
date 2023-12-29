"""
Monte Carlo GLIE Control Algorithm
- Stochastic Policy 
- Policy probabilities are updated during the learning process.
"""
import numpy as np
from collections import defaultdict
import sys
from plot_blackjack import *


class MCAgent():
  def __init__(self, env, alpha=0.0001, gamma=0.99):
    self.env = env
    self.alpha = alpha    # learning rate
    self.gamma = gamma    # discount factor
    self.n_action = self.env.action_space.n
    print('Problem being solved by the agent: ', self.env.spec.name)

  def get_policy_probs(self, Q_s, epsilon):
    """
    Get the probability of taking the best known action according to the value of epsilon.
    Returns the policy for the given Q value.
    """
    nA = self.n_action
    policy_s = np.ones(nA) * epsilon / nA
    best_a = np.argmax(Q_s)
    policy_s[best_a] = 1 - epsilon + (epsilon / nA) # increase the probability of best action
    return policy_s

  def best_policy(self, Q):
    """
    returns the best action for each Q value in the policy
    """
    return dict((k, np.argmax(v)) for k, v in Q.items())

  def update_Q(self, episode, Q):
    """
    Update the Q value using the formula:
    Q = Q + alpha * (G - Q)
    G: cumulative discounted reward for the current episode
    alpha: learning rate
    """
    for s, a, r in episode:
      first_occurrence_idx = next(i for i, x in enumerate(episode) if x[0] == s)
      G = sum([x[2] * (self.gamma ** i) for i, x in enumerate(episode[first_occurrence_idx:])])
      Q[s][a] += self.alpha * (G - Q[s][a])
    return Q

  def generate_episode(self, Q, epsilon):
    """
    Generate episode with the given Q value
    """
    nA = self.n_action
    episode = []
    state = self.env.reset()[0]
    rewards = 0
    while True:
      probs = self.get_policy_probs(Q[state], epsilon)
      action = np.random.choice(np.arange(self.n_action), p=probs)\
          if state in Q else env.action_space.sample()    # epsilon-greedy policy
      next_state, reward, done, info, _ = env.step(action)
      rewards += reward
      episode.append((state, action, reward))
      state = next_state
      if done:
        break
    return episode, rewards

  def mc_control(self, num_episodes=500000):
    """
    main method:
    - iterates through episodes to update Q value and update epsilon.
    """
    epsilon = 1.0
    eps_min = 0.01
    decay = 0.9999

    nA = self.env.action_space.n
    Q = defaultdict(lambda: np.zeros(nA))
    ep_rewards = []
    for i in range(1, num_episodes+1):
      epsilon = max(epsilon * decay, eps_min)
      episode, rewards = self.generate_episode(Q, epsilon)
      Q = self.update_Q(episode, Q)
      ep_rewards.append(rewards)
      if i % 1000 == 0:
        print('\rEpisode {}/{}, Episodic reward: {}'.format(i, num_episodes, np.mean(ep_rewards)), end="")
        sys.stdout.flush()
    policy = self.best_policy(Q)
    return policy, Q

  def validate(self, policy=None, num_episodes=10):
    ep_rewards, ep_steps = [], []
    #ipdb.set_trace()
    for i in range(num_episodes):
      #print('\nepisode: ', i)
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
  
if __name__ == '__main__':
    import gymnasium as gym

    # Create an environment
    env = gym.make('Blackjack-v1')

    # Create a Monte-Carlo agent
    agent = MCAgent(env)

    # Learn optimal policy
    policy, Q = agent.mc_control(num_episodes=500000)

    # Compute value function
    V = dict((k, np.max(v)) for k, v in Q.items())

    plot_blackjack_policy(policy)
    plot_blackjack_values(V)

