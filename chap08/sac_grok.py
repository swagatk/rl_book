

import gymnasium as gym
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from collections import deque
import random

# Set random seeds for reproducibility
tf.random.set_seed(42)
np.random.seed(42)

# Replay Buffer
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        state, action, reward, next_state, done = zip(*random.sample(self.buffer, batch_size))
        return (np.array(state), np.array(action), np.array(reward, dtype=np.float32),
                np.array(next_state), np.array(done, dtype=np.float32))
    
    def __len__(self):
        return len(self.buffer)

# Actor Network
class Actor(tf.keras.Model):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()
        self.dense1 = tf.keras.layers.Dense(256, activation='relu')
        self.dense2 = tf.keras.layers.Dense(256, activation='relu')
        self.mean = tf.keras.layers.Dense(action_dim)
        self.log_std = tf.keras.layers.Dense(action_dim)
        self.max_action = max_action
    
    def call(self, state):
        x = self.dense1(state)
        x = self.dense2(x)
        mean = self.mean(x)
        log_std = self.log_std(x)
        log_std = tf.clip_by_value(log_std, -20, 2)  # Stabilize log_std
        return mean, log_std
    
    def sample_action(self, state, reparameterize=False):
        mean, log_std = self.call(state)
        std = tf.exp(log_std)
        dist = tfp.distributions.Normal(mean, std)
        if reparameterize:
            z = mean + std * tf.random.normal(tf.shape(mean))
        else:
            z = dist.sample()
        action = tf.tanh(z) * self.max_action
        log_prob = dist.log_prob(z) - tf.reduce_sum(tf.math.log(1 - tf.tanh(z)**2 + 1e-6), axis=-1, keepdims=True)
        return action, log_prob

# Critic Network
class Critic(tf.keras.Model):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.dense1 = tf.keras.layers.Dense(256, activation='relu')
        self.dense2 = tf.keras.layers.Dense(256, activation='relu')
        self.q = tf.keras.layers.Dense(1)
    
    def call(self, state, action):
        x = tf.concat([state, action], axis=-1)
        x = self.dense1(x)
        x = self.dense2(x)
        return self.q(x)

class ValueNetwork(tf.keras.Model):
    def __init__(self, state_dim):
        super(ValueNetwork, self).__init__()
        self.dense1 = tf.keras.layers.Dense(256, activation='relu')
        self.dense2 = tf.keras.layers.Dense(256, activation='relu')
        self.value = tf.keras.layers.Dense(1)
    
    def call(self, state):
        x = self.dense1(state)
        x = self.dense2(x)
        return self.value(x)

# SAC Agent
class SAC:
    def __init__(self, state_dim, action_dim, max_action):
        self.actor = Actor(state_dim, action_dim, max_action)
        self.critic_1 = Critic(state_dim, action_dim)
        self.critic_2 = Critic(state_dim, action_dim)
        self.value = ValueNetwork(state_dim)
        self.target_value = ValueNetwork(state_dim)
        
        self.actor_optimizer = tf.keras.optimizers.Adam(learning_rate=3e-4)
        self.critic_1_optimizer = tf.keras.optimizers.Adam(learning_rate=3e-4)
        self.critic_2_optimizer = tf.keras.optimizers.Adam(learning_rate=3e-4)
        self.value_optimizer = tf.keras.optimizers.Adam(learning_rate=3e-4)
        self.log_alpha = tf.Variable(0.0, trainable=True)
        self.alpha_optimizer = tf.keras.optimizers.Adam(learning_rate=3e-4)
        
        self.target_entropy = -action_dim  # Target entropy = -|A|
        self.gamma = 0.99
        self.tau = 0.005
        self.max_action = max_action
        
        # Initialize target value network
        self.target_value.set_weights(self.value.get_weights())
    
    def update_target(self):
        for target, source in zip(self.target_value.weights, self.value.weights):
            target.assign(self.tau * source + (1.0 - self.tau) * target)
    
    def train_step(self, states, actions, rewards, next_states, dones, batch_size):
        alpha = tf.exp(self.log_alpha)
        
        # Value Update
        with tf.GradientTape() as tape:
            next_actions, next_log_probs = self.actor.sample_action(next_states, reparameterize=False)
            next_q1 = self.critic_1(next_states, next_actions)
            next_q2 = self.critic_2(next_states, next_actions)
            next_q = tf.minimum(next_q1, next_q2)
            target_v = next_q - alpha * next_log_probs
            v = self.value(states)
            value_loss = tf.reduce_mean(tf.square(target_v - v))
        
        value_grads = tape.gradient(value_loss, self.value.trainable_variables)
        self.value_optimizer.apply_gradients(zip(value_grads, self.value.trainable_variables))
        
        # Critic Update
        with tf.GradientTape(persistent=True) as tape:
            q1 = self.critic_1(states, actions)
            q2 = self.critic_2(states, actions)
            next_v = self.target_value(next_states)
            target_q = rewards + self.gamma * (1 - dones) * next_v
            critic_1_loss = tf.reduce_mean(tf.square(target_q - q1))
            critic_2_loss = tf.reduce_mean(tf.square(target_q - q2))
        
        critic_1_grads = tape.gradient(critic_1_loss, self.critic_1.trainable_variables)
        critic_2_grads = tape.gradient(critic_2_loss, self.critic_2.trainable_variables)
        self.critic_1_optimizer.apply_gradients(zip(critic_1_grads, self.critic_1.trainable_variables))
        self.critic_2_optimizer.apply_gradients(zip(critic_2_grads, self.critic_2.trainable_variables))
        del tape
        
        # Actor Update
        with tf.GradientTape() as tape:
            new_actions, log_probs = self.actor.sample_action(states, reparameterize=True)
            q1_new = self.critic_1(states, new_actions)
            q2_new = self.critic_2(states, new_actions)
            q_new = tf.minimum(q1_new, q2_new)
            actor_loss = tf.reduce_mean(alpha * log_probs - q_new)
        
        actor_grads = tape.gradient(actor_loss, self.actor.trainable_variables)
        self.actor_optimizer.apply_gradients(zip(actor_grads, self.actor.trainable_variables))
        
        # Alpha Update
        with tf.GradientTape() as tape:
            _, log_probs = self.actor.sample_action(states, reparameterize=True)
            alpha_loss = tf.reduce_mean(tf.negative(self.log_alpha) * (log_probs + self.target_entropy))
        
        alpha_grads = tape.gradient(alpha_loss, [self.log_alpha])
        self.alpha_optimizer.apply_gradients(zip(alpha_grads, [self.log_alpha]))
        
        self.update_target()
        
        return value_loss, critic_1_loss, critic_2_loss, actor_loss, alpha_loss

def main():
    # Environment setup
    env = gym.make("LunarLanderContinuous-v3")
    state_dim = env.observation_space.shape[0]  # 8
    action_dim = env.action_space.shape[0]      # 2
    max_action = float(env.action_space.high[0]) # 1.0
    
    # SAC agent and replay buffer
    agent = SAC(state_dim, action_dim, max_action)
    replay_buffer = ReplayBuffer(capacity=1000000)
    batch_size = 256
    max_episodes = 1000
    max_steps = 1000
    
    # Training loop
    total_rewards = []
    for episode in range(max_episodes):
        state, _ = env.reset()
        episode_reward = 0
        done = False
        step = 0
        
        while not done and step < max_steps:
            state_tf = tf.expand_dims(tf.convert_to_tensor(state, dtype=tf.float32), 0)
            action, _ = agent.actor.sample_action(state_tf)
            action = action.numpy()[0]
            
            next_state, reward, done, truncated, _ = env.step(action)
            done = done or truncated
            
            replay_buffer.push(state, action, reward, next_state, done)
            state = next_state
            episode_reward += reward
            step += 1
            
            if len(replay_buffer) > batch_size:
                batch = replay_buffer.sample(batch_size)
                states, actions, rewards, next_states, dones = [tf.convert_to_tensor(x, dtype=tf.float32) for x in batch]
                losses = agent.train_step(states, actions, rewards, next_states, dones, batch_size)
        
        total_rewards.append(episode_reward)
        if episode % 10 == 0:
            print(f"Episode {episode}, Reward: {episode_reward:.2f}, Avg Reward: {np.mean(total_rewards[-10:]):.2f}")
        
        # Early stopping if solved (average reward > 200)
        if len(total_rewards) >= 100 and np.mean(total_rewards[-100:]) > 200:
            print(f"Solved at episode {episode}!")
            break
    
    env.close()

if __name__ == "__main__":
    main()

