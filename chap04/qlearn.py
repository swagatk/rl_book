import gymnasium as gym
import numpy as np
import sys 
import time

class QLearningAgent():
  def __init__(self, env, alpha=0.3, gamma=0.99, fixed_epsilon=None):
    self.env = env
    self.alpha = alpha
    self.gamma = gamma
    if fixed_epsilon is None:
      self.epsilon = 1.0
      self.eps_min = 0.01
      self.decay_rate = 0.999
      self.decay_flag = True 
    else:
      self.epsilon = fixed_epsilon
      self.decay_flag = False
  
    print('Environment Name: ', self.env.spec.name)
    print('RL Agent: ', 'Q-Learning')

    # initialize Q table
    self.Q = np.zeros((self.env.observation_space.n, self.env.action_space.n))


  def epsilon_greedy_policy(self, state, epsilon):
    randvar = np.random.uniform(0, 1)
    if randvar < epsilon:
      action = self.env.action_space.sample() # explore
    else:
      if np.max(self.Q[state]) > 0:
        action = np.argmax(self.Q[state]) # exploit
      else:
        action = self.env.action_space.sample() # explore
    return action


  def update_q_table(self, s, a, r, s_next):
    self.Q[s][a] += self.alpha * (r + self.gamma * \
                                  np.max(self.Q[s_next]) - self.Q[s][a])
  
  def train(self, num_episodes=1000, filename=None, freq=100):
    if filename is not None:
      file = open(filename, "w")
 
    ep_rewards = []
    start = time.time()
    for i in range(num_episodes):
      ep_reward = 0
      # reset the environment for each episode
      state = self.env.reset()[0]
      while True:
        # select an action
        action = self.epsilon_greedy_policy(state, self.epsilon)
        # obtain rewards
        next_state, reward, done, _, _ = self.env.step(action)
        # update q table
        self.update_q_table(state, action, reward, next_state)
        # accumulate rewards for the episode
        ep_reward += reward
        # prepare for next iteration
        state = next_state
        if done: # end of episode
          ep_rewards.append(ep_reward)
          break
      #end of while loop
      if self.decay_flag: # allow epsilon decay
        self.epsilon = max(self.epsilon * self.decay_rate, self.eps_min)

      if filename is not None:
        file.write("{}\t{}\n".format(np.mean(ep_rewards), self.epsilon))
      if i % freq == 0:
        print('\rEpisode: {}/{}, Average episodic Reward:{:.3f}.'\
              .format(i, num_episodes, np.mean(ep_rewards)), end="")
        sys.stdout.flush()
    #end of for loop
    end = time.time()
    print('\nTraining time (seconds): ', (end - start))
    if filename is not None:
      file.close()

  def validate(self, num_episodes=10):
    ep_rewards = []
    for i in range(num_episodes):
      state = self.env.reset()[0]
      ep_reward = 0
      while True:
        action = self.epsilon_greedy_policy(state, epsilon=0)
        next_state, reward, done, _, _ = self.env.step(action)
        ep_reward += reward
        state = next_state 
        if done:
          ep_rewards.append(ep_reward)
          break
    print('\nTest: Average Episodic Reward: ', np.mean(ep_rewards))

  def display_q_table(self):
    print("\n ------------------ \n")
    print(self.Q)
    print("\n ------------------ \n")

  def __delete__(self):
    self.env.close()


if __name__ == '__main__':
  
    #env = gym.make('Taxi-v3')
    env = gym.make('FrozenLake-v1', is_slippery=False)
    agent = QLearningAgent(env, alpha=0.1)
    agent.train(num_episodes=1000, filename='flake_qlearn.tsv', freq=100)
    agent.validate()