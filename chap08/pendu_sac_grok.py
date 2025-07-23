import gymnasium as gym
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import collections
import random

# Hyperparameters
GAMMA = 0.99
TAU = 0.005
ALPHA = 0.2
ACTOR_LR = 3e-4
CRITIC_LR = 3e-4
BATCH_SIZE = 256
BUFFER_SIZE = 1000000
TARGET_UPDATE_INTERVAL = 1
MAX_EPISODES = 200
MAX_STEPS = 200

# Replay Buffer
class ReplayBuffer:
    def __init__(self, buffer_size, state_shape, action_dim):
        self.buffer = collections.deque(maxlen=buffer_size)
        self.state_shape = state_shape
        self.action_dim = action_dim

    def append(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards, dtype=np.float32),
            np.array(next_states),
            np.array(dones, dtype=np.float32)
        )

    def __len__(self):
        return len(self.buffer)

# Actor Network
class Actor(tf.keras.Model):
    def __init__(self, action_dim, max_action):
        super(Actor, self).__init__()
        self.d1 = layers.Dense(256, activation='relu')
        self.d2 = layers.Dense(256, activation='relu')
        self.mu = layers.Dense(action_dim)
        self.log_std = layers.Dense(action_dim)
        self.max_action = max_action

    def call(self, state):
        x = self.d1(state)
        x = self.d2(x)
        mu = self.mu(x)
        log_std = self.log_std(x)
        log_std = tf.clip_by_value(log_std, -20, 2)
        std = tf.exp(log_std)
        return mu, std

    def sample_action(self, state):
        mu, std = self.call(state)
        dist = tf.random.normal(tf.shape(mu), mean=mu, stddev=std)
        action = tf.tanh(dist) * self.max_action
        log_prob = self._log_prob(dist, mu, std)
        return action, log_prob

    def _log_prob(self, sample, mu, std):
        log_prob = -0.5 * tf.reduce_sum(
            tf.math.log(2 * np.pi * tf.square(std)) + tf.square((sample - mu) / (std + 1e-6)), axis=-1)
        log_prob -= tf.reduce_sum(2 * tf.math.log(1 - tf.square(tf.tanh(sample)) + 1e-6), axis=-1)
        return log_prob

# Critic Network
class Critic(tf.keras.Model):
    def __init__(self):
        super(Critic, self).__init__()
        self.d1 = layers.Dense(256, activation='relu')
        self.d2 = layers.Dense(256, activation='relu')
        self.out = layers.Dense(1)

    def call(self, state, action):
        x = tf.concat([state, action], axis=-1)
        x = self.d1(x)
        x = self.d2(x)
        return self.out(x)

# SAC Agent
class SAC:
    def __init__(self, state_shape, action_dim, max_action):
        self.actor = Actor(action_dim, max_action)
        self.critic_1 = Critic()
        self.critic_2 = Critic()
        self.target_critic_1 = Critic()
        self.target_critic_2 = Critic()
        self.actor_optimizer = tf.keras.optimizers.Adam(ACTOR_LR)
        self.critic_1_optimizer = tf.keras.optimizers.Adam(CRITIC_LR)
        self.critic_2_optimizer = tf.keras.optimizers.Adam(CRITIC_LR)
        self.replay_buffer = ReplayBuffer(BUFFER_SIZE, state_shape, action_dim)
        self.max_action = max_action
        self._update_target_networks(tau=1.0)

    def _update_target_networks(self, tau):
        for target, source in [
            (self.target_critic_1, self.critic_1),
            (self.target_critic_2, self.critic_2)
        ]:
            for target_weights, source_weights in zip(target.trainable_variables, source.trainable_variables):
                target_weights.assign(tau * source_weights + (1 - tau) * target_weights)

    def train_step(self):
        if len(self.replay_buffer) < BATCH_SIZE:
            return

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(BATCH_SIZE)

        with tf.GradientTape(persistent=True) as tape:
            next_actions, next_log_probs = self.actor.sample_action(next_states)
            q1_next = self.target_critic_1(next_states, next_actions)
            q2_next = self.target_critic_2(next_states, next_actions)
            q_next = tf.minimum(q1_next, q2_next) - ALPHA * next_log_probs
            target_q = rewards + GAMMA * (1 - dones) * q_next

            q1 = self.critic_1(states, actions)
            q2 = self.critic_2(states, actions)
            critic_1_loss = tf.reduce_mean(tf.square(target_q - q1))
            critic_2_loss = tf.reduce_mean(tf.square(target_q - q2))

            new_actions, log_probs = self.actor.sample_action(states)
            q1_pi = self.critic_1(states, new_actions)
            q2_pi = self.critic_2(states, new_actions)
            actor_loss = tf.reduce_mean(ALPHA * log_probs - tf.minimum(q1_pi, q2_pi))

        critic_1_grads = tape.gradient(critic_1_loss, self.critic_1.trainable_variables)
        self.critic_1_optimizer.apply_gradients(zip(critic_1_grads, self.critic_1.trainable_variables))
        critic_2_grads = tape.gradient(critic_2_loss, self.critic_2.trainable_variables)
        self.critic_2_optimizer.apply_gradients(zip(critic_2_grads, self.critic_2.trainable_variables))
        actor_grads = tape.gradient(actor_loss, self.actor.trainable_variables)
        self.actor_optimizer.apply_gradients(zip(actor_grads, self.actor.trainable_variables))

    def act(self, state):
        state = tf.expand_dims(tf.convert_to_tensor(state, dtype=tf.float32), axis=0)
        action, _ = self.actor.sample_action(state)
        return action.numpy()[0]

def main():
    env = gym.make('Pendulum-v1')
    state_shape = env.observation_space.shape
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])
    agent = SAC(state_shape, action_dim, max_action)

    scores = []
    for episode in range(MAX_EPISODES):
        state = env.reset()[0]
        episode_reward = 0

        for step in range(MAX_STEPS):
            action = agent.act(state)
            next_state, reward, done, _, _ = env.step(action)
            agent.replay_buffer.append(state, action, reward, next_state, done)
            state = next_state
            episode_reward += reward

            agent.train_step()
            if step % TARGET_UPDATE_INTERVAL == 0:
                agent._update_target_networks(TAU)

            if done:
                scores.append(episode_reward)
                break

        print(f"Episode {episode + 1}, Reward: {episode_reward:.2f}, \
              average score: {np.mean(scores):.2f} ")

if __name__ == "__main__":
    main()