import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import gymnasium as gym
import collections
import wandb

# --- Configuration ---
H_SIZE = 256        # Deterministic state size (RNN)
Z_SIZE = 32         # Stochastic state size
BATCH_SIZE = 16
SEQ_LEN = 25        # Length of sequences for world model training
HORIZON = 10        # How far to "dream" in the latent space
LR_MODEL = 3e-4
LR_AGENT = 1e-4

# --- 1. World Model Components ---

class RSSM(tf.keras.Model):
    """
    Recurrent State Space Model (RSSM).
    Combines a deterministic RNN state (h) with a stochastic latent state (z).
    """
    def __init__(self, h_size, z_size):
        super().__init__()
        self.h_size = h_size
        self.z_size = z_size
        self.cell = layers.GRUCell(h_size)
        
        # Mapping (h_t, z_{t-1}, a_{t-1}) -> h_t
        self.obs_embed = layers.Dense(h_size, activation='relu')
        
        # Stochastic state dynamics (Prior): p(z_t | h_t)
        self.prior_net = layers.Dense(z_size * 2) # Mean and StdDev
        
        # Stochastic state posterior (Posterior): q(z_t | h_t, e_t)
        # e_t is the embedding of the actual observation
        self.post_net = layers.Dense(z_size * 2)

    def get_initial_state(self, batch_size):
        return (tf.zeros([batch_size, self.h_size]), 
                tf.zeros([batch_size, self.z_size]))

    def observe(self, embed, action, state):
        """Used during world model training with real data."""
        h, z = state
        h, _ = self.cell(tf.concat([z, action], axis=-1), [h])
        
        # Posterior: q(z | h, e)
        post_out = self.post_net(tf.concat([h, embed], axis=-1))
        mean, std = tf.split(post_out, 2, axis=-1)
        std = tf.nn.softplus(std) + 0.1
        z = mean + std * tf.random.normal(tf.shape(mean))
        return (h, z), (mean, std)

    def imagine(self, action, state):
        """Used during dreaming/planning."""
        h, z = state
        h, _ = self.cell(tf.concat([z, action], axis=-1), [h])
        
        # Prior: p(z | h)
        prior_out = self.prior_net(h)
        mean, std = tf.split(prior_out, 2, axis=-1)
        std = tf.nn.softplus(std) + 0.1
        z = mean + std * tf.random.normal(tf.shape(mean))
        return (h, z), (mean, std)

# --- 2. Encoder, Decoder, Reward ---

def build_encoder(obs_shape):
    return tf.keras.Sequential([
        layers.InputLayer(input_shape=obs_shape),
        layers.Dense(256, activation='relu'),
        layers.Dense(256, activation='relu'),
    ])

def build_decoder(obs_shape, h_size, z_size):
    return tf.keras.Sequential([
        layers.InputLayer(input_shape=(h_size + z_size,)),
        layers.Dense(256, activation='relu'),
        layers.Dense(256, activation='relu'),
        layers.Dense(obs_shape[0]) # Reconstruct state vector
    ])

def build_reward_model(h_size, z_size):
    return tf.keras.Sequential([
        layers.InputLayer(input_shape=(h_size + z_size,)),
        layers.Dense(256, activation='relu'),
        layers.Dense(1)
    ])

# --- 3. The Actor-Critic Agent (Operating in Imagination) ---

class Actor(tf.keras.Model):
    def __init__(self, action_dim):
        super().__init__()
        self.net = tf.keras.Sequential([
            layers.Dense(256, activation='relu'),
            layers.Dense(256, activation='relu'),
            layers.Dense(action_dim, activation='tanh') # Continuous control
        ])
    def call(self, latent):
        return self.net(latent)

class Critic(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.net = tf.keras.Sequential([
            layers.Dense(256, activation='relu'),
            layers.Dense(256, activation='relu'),
            layers.Dense(1)
        ])
    def call(self, latent):
        return self.net(latent)

# --- 4. Dreamer Agent Implementation ---

class DreamerV1:
    def __init__(self, obs_shape, action_dim):
        self.obs_shape = obs_shape
        self.action_dim = action_dim
        
        # World Model
        self.rssm = RSSM(H_SIZE, Z_SIZE)
        self.encoder = build_encoder(obs_shape)
        self.decoder = build_decoder(obs_shape, H_SIZE, Z_SIZE)
        self.reward_model = build_reward_model(H_SIZE, Z_SIZE)
        
        # Agent
        self.actor = Actor(action_dim)
        self.critic = Critic()
        
        self.model_opt = tf.keras.optimizers.Adam(LR_MODEL)
        self.agent_opt = tf.keras.optimizers.Adam(LR_AGENT)
        
        self.buffer = collections.deque(maxlen=100000)

    def select_action(self, obs, state, training=True):
        """Action selection using the current observation and RSSM state."""
        obs = tf.convert_to_tensor(obs[None], dtype=tf.float32)
        embed = self.encoder(obs)
        
        action_placeholder = tf.zeros([1, self.action_dim]) # First step has no prev action
        state, _ = self.rssm.observe(embed, action_placeholder, state)
        
        h, z = state
        latent = tf.concat([h, z], axis=-1)
        action = self.actor(latent)
        
        if training:
            action += tf.random.normal(tf.shape(action)) * 0.1 # Exploration noise
        
        return tf.clip_by_value(action[0], -1.0, 1.0), state

    def train(self):
        if len(self.buffer) < BATCH_SIZE * SEQ_LEN:
            return

        # 1. Sample sequences from buffer
        indices = np.random.randint(0, len(self.buffer) - SEQ_LEN, BATCH_SIZE)
        obs_seq, act_seq, rew_seq = [], [], []
        for idx in indices:
            seq = list(self.buffer)[idx:idx+SEQ_LEN]
            obs_seq.append([s[0] for s in seq])
            act_seq.append([s[1] for s in seq])
            rew_seq.append([s[2] for s in seq])
            
        obs_seq = tf.convert_to_tensor(obs_seq, dtype=tf.float32)
        act_seq = tf.convert_to_tensor(act_seq, dtype=tf.float32)
        rew_seq = tf.convert_to_tensor(rew_seq, dtype=tf.float32)

        # 2. Train World Model
        with tf.GradientTape() as model_tape:
            state = self.rssm.get_initial_state(BATCH_SIZE)
            
            enc_obs = self.encoder(tf.reshape(obs_seq, [-1, *self.obs_shape]))
            enc_obs = tf.reshape(enc_obs, [BATCH_SIZE, SEQ_LEN, -1])
            
            h_seq, z_seq = [], []
            kl_loss = 0.0
            
            for t in range(SEQ_LEN):
                embed = enc_obs[:, t]
                action = act_seq[:, t]
                
                # Manual step inside tape
                h, z_prev = state
                h, _ = self.rssm.cell(tf.concat([z_prev, action], axis=-1), [h])
                
                # Prior
                prior_out = self.rssm.prior_net(h)
                prior_mean, prior_std = tf.split(prior_out, 2, axis=-1)
                prior_std = tf.nn.softplus(prior_std) + 0.1
                
                # Posterior
                post_out = self.rssm.post_net(tf.concat([h, embed], axis=-1))
                post_mean, post_std = tf.split(post_out, 2, axis=-1)
                post_std = tf.nn.softplus(post_std) + 0.1
                
                z = post_mean + post_std * tf.random.normal(tf.shape(post_mean))
                state = (h, z)
                
                h_seq.append(h)
                z_seq.append(z)
                
                # KL loss between posterior and prior (simplified diagonal Gaussian KL)
                var1, var2 = tf.square(post_std), tf.square(prior_std)
                kl = tf.math.log(prior_std / post_std) + (var1 + tf.square(post_mean - prior_mean)) / (2.0 * var2) - 0.5
                kl_loss += tf.reduce_mean(kl)

            h_seq = tf.stack(h_seq, axis=1)
            z_seq = tf.stack(z_seq, axis=1)
            latent_seq = tf.concat([h_seq, z_seq], axis=-1)
            
            latent_flat = tf.reshape(latent_seq, [-1, H_SIZE + Z_SIZE])
            
            # Reconstruction and Reward losses
            rec_obs = self.decoder(latent_flat)
            rec_obs = tf.reshape(rec_obs, [BATCH_SIZE, SEQ_LEN, *self.obs_shape])
            rec_loss = tf.reduce_mean(tf.square(rec_obs - obs_seq))
            
            pred_rew = self.reward_model(latent_flat)
            pred_rew = tf.reshape(pred_rew, [BATCH_SIZE, SEQ_LEN])
            rew_loss = tf.reduce_mean(tf.square(pred_rew - rew_seq))
            
            model_loss = rec_loss + rew_loss + kl_loss

        # 3. Dreaming & Agent Training
        with tf.GradientTape(persistent=True) as agent_tape:
            # Flatten to dream from all points simultaneously
            h_flat = tf.reshape(h_seq, [-1, H_SIZE])
            z_flat = tf.reshape(z_seq, [-1, Z_SIZE])
            
            # Stop gradients from flowing back to world model during agent training
            state = (tf.stop_gradient(h_flat), tf.stop_gradient(z_flat))
            
            rewards, values = [], []
            
            for _ in range(HORIZON):
                h, z = state
                latent = tf.concat([h, z], axis=-1)
                action = self.actor(latent)
                
                # Imagine next state
                state, _ = self.rssm.imagine(action, state)
                h_next, z_next = state
                latent_next = tf.concat([h_next, z_next], axis=-1)
                
                reward = self.reward_model(latent_next)
                value = self.critic(latent_next)
                
                rewards.append(reward)
                values.append(value)
                
            rewards = tf.stack(rewards, axis=0) # [HORIZON, BATCH*SEQ_LEN, 1]
            values = tf.stack(values, axis=0)
            
            # Value targets
            discounts = 0.99
            returns = rewards[:-1] + discounts * values[1:]
            
            # Actor Loss: Maximize expected returns (we minimize negative returns)
            actor_loss = -tf.reduce_mean(values)
            
            # Critic Loss: Minimize regression error against targets
            critic_loss = tf.reduce_mean(tf.square(values[:-1] - tf.stop_gradient(returns)))
            
            agent_loss = actor_loss + critic_loss

        # Apply gradients
        model_vars = self.encoder.trainable_variables + self.decoder.trainable_variables + \
                     self.rssm.trainable_variables + self.reward_model.trainable_variables
        agent_vars = self.actor.trainable_variables + self.critic.trainable_variables
        
        model_grads = model_tape.gradient(model_loss, model_vars)
        agent_grads = agent_tape.gradient(agent_loss, agent_vars)
        
        self.model_opt.apply_gradients(zip(model_grads, model_vars))
        self.agent_opt.apply_gradients(zip(agent_grads, agent_vars))
        
        del agent_tape

# --- Main Loop ---
if __name__ == "__main__":
    # Initialize wandb
    wandb.init(
        project="dreamer-v1-lunar-lander",
        config={
            "h_size": H_SIZE,
            "z_size": Z_SIZE,
            "batch_size": BATCH_SIZE,
            "seq_len": SEQ_LEN,
            "horizon": HORIZON,
            "lr_model": LR_MODEL,
            "lr_agent": LR_AGENT,
            "env": "LunarLanderContinuous-v3"
        }
    )

    env = gym.make("LunarLanderContinuous-v3")
    agent = DreamerV1(env.observation_space.shape, env.action_space.shape[0])
    
    reward_history = collections.deque(maxlen=100)
    total_rewards_sum = 0
    
    for episode in range(1, 101):
        obs, _ = env.reset()
        state = agent.rssm.get_initial_state(1)
        total_reward = 0
        
        for step in range(500):
            action, state = agent.select_action(obs, state)
            next_obs, reward, done, trunc, _ = env.step(action.numpy())
            
            # Store in buffer
            agent.buffer.append((obs, action, reward, next_obs, done))
            
            obs = next_obs
            total_reward += reward
            
            if (step + 1) % 50 == 0:
                agent.train()
                
            if done or trunc:
                break
        
        # Track metrics
        reward_history.append(total_reward)
        total_rewards_sum += total_reward
        avg_reward = total_rewards_sum / episode
        last_100_avg = np.mean(reward_history)

        # Log to wandb
        wandb.log({
            "episodic_reward": total_reward,
            "average_episodic_reward": avg_reward,
            "last_100_episodes_average": last_100_avg,
            "episode": episode
        })
        
        print(f"Episode {episode}: Total Reward {total_reward:.2f} | Avg (100): {last_100_avg:.2f}")

    wandb.finish()