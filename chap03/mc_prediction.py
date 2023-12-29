from collections import defaultdict
import numpy as np
import random
import gymnasium as gym
import sys 
import matplotlib.pyplot as plt
from plot_blackjack import plot_blackjack_values

class MCPAgent:
  def __init__(self, env, gamma=0.99):
    self.env = env
    self.n_actions = env.action_space.n
    self.gamma = gamma
    print('Environment: ', self.env.spec.name)

  def sample_policy(self, obs):
    player_sum, dealer_show, usable_ace = obs
    probs = [0.8, 0.2] if player_sum > 18 else [0.2, 0.8]
    action = np.random.choice(np.arange(2), p=probs)
    return action

  def generate_episode(self):
    """
    Plays a single episode with a set policy in the environment given. Records the state, action
    and reward for each step and returns the all timesteps for the episode.
    """
    episode = []
    state = env.reset()[0]
    while True:
      action = self.sample_policy(state)
      next_state, reward, done, info, _ = env.step(action)
      episode.append((state, action, reward))
      state = next_state
      if done:
        break
    return episode

  def update_Q_first_visit(self, episode, Q, returns_sum, N, method=1):
    """
    Implements First Visit Monte-Carlo update of Q values.
    Args:
      episode: state, action, reward : tuple
      N: first-visit count of each state action pair: dict
      return_sum: returns/sum of returns for each state-action pair: dict
      Q: Q-value for each state action pair: dict
      method: one of the two implementations
    Output:
      Q, N, return_sum are modified.
    """
    for s, a, r in episode:
      first_occurrence_idx = next(i \
              for i, x in enumerate(episode) if x[0] == s)
      G = sum([x[2] * (self.gamma ** i) \
              for i, x in enumerate(episode[first_occurrence_idx:])])
      N[s][a] += 1  # first visit count for (s,a)
      if method == 1:
        returns_sum[s][a] += G
        Q[s][a] = returns_sum[s][a] / N[s][a]
      else:
        returns_sum[s][a] = G
        Q[s][a] += (returns_sum[s][a] - Q[s][a]) / N[s][a]

  def update_Q_first_visit_2(self, episode, Q, returns_sum, N, method=1):
    G = 0
    visited_state = set()
    for i in range(len(episode)-1, -1, -1):    # traverse in reverse order
      state, action, reward = episode[i]
      G += self.gamma ** i * reward # returns for (s,a)
      if (state, action) not in visited_state:    # if first visit
        N[state][action] += 1  # first visit count
        if method == 1:
          returns_sum[state][action] += G  # update returns
          Q[state][action] = returns_sum[state][action] / N[state][action]
        else:
          returns_sum[state][action] = G # update returns
          Q[state][action] += (returns_sum[state][action] -
                               Q[state][action])/ N[state][action]
        visited_state.add((state, action))

  def update_Q_every_visit(self, episode, Q, returns_sum, N, method=1):
    G = 0  # return
    for i in range(len(episode)-1, -1, -1):
      s, a, r = episode[i]
      G += self.gamma ** i * r  # returns for (s,a)
      N[s][a] += 1    # every-visit count
      if method == 1:
        returns_sum[s][a] += G
        Q[s][a] = returns_sum[s][a] / N[s][a]
      else:
        returns_sum[s][a] = G
        Q[s][a] += (returns_sum[s][a] - Q[s][a]) / N[s][a]


  def mc_predict_2(self, num_episodes=1000000):
    Q = defaultdict(lambda: np.zeros(self.n_actions))  # Q(s,a)
    N = defaultdict(lambda: np.zeros(self.n_actions))  # N(s,a)
    returns_sum = defaultdict(lambda: np.zeros(self.n_actions))
    for ep in range(num_episodes):
      episode = self.generate_episode()
      self.update_Q_first_visit_2(episode, Q, returns_sum, N)
      if ep % 1000 == 0 and ep != 0:
        print("\rEpisode: {}/{}".format(ep, num_episodes), end="")
        sys.stdout.flush()
    # printing random visit counts for various (s,a) pairs
    # for _ in range(10):
    #   print(random.choice(list(returns.values())))
    return Q

  def mc_predict(self, num_episodes=100000, first_visit=False, method=1):
    """ This plays through several episodes of the game """
    returns_sum = defaultdict(lambda: np.zeros(self.env.action_space.n))
    N = defaultdict(lambda: np.zeros(self.env.action_space.n))
    Q = defaultdict(lambda: np.zeros(self.env.action_space.n))
    for i in range(1, num_episodes+1):
      if i % 1000 == 0:
        print('\rEpisode: {}/{}.'.format(i, num_episodes), end="")
        sys.stdout.flush()
      episode = self.generate_episode()
      if first_visit:
        self.update_Q_first_visit(episode, Q, returns_sum, N, method)
      else:
        self.update_Q_every_visit(episode, Q, returns_sum, N, method)
    return Q

  def QtoVwProb(self, Q):
    '''
    Converts Q to V
    Args:
      Q(s,a): dict
    '''
    V = dict((k, (k[0]>18) * (np.dot([0.8, 0.2], v)) + \
     (k[0] <= 18) * (np.dot([0.2, 0.8], v))) for k, v in Q.items())
    return V


if __name__ == '__main__':
    # create gym environment
    env = gym.make('Blackjack-v1', render_mode="rgb_array")  
    agent = MCPAgent(env)
    Q = agent.mc_predict(num_episodes=500000, first_visit=False, method=1)
    #Q = agent.mc_predict_2(num_episodes=500000)  # same effect no change
    V = agent.QtoVwProb(Q)
    plot_blackjack_values(V)