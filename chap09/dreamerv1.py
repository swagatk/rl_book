import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import gymnasium as gym
import collections
import wandb



# --- 1. World Model Components ---

class RSSM(tf.keras.Model):
    def __init__(self, h_size, z_size):
        super().__init__()
        self.h_size = h_size
        self.z_size = z_size
        self.cell = layers.GRUCell(h_size)
        self.obs_embed = layers.Dense(h_size, activation='relu')
        self.prior_net = layers.Dense(z_size * 2)
        self.post_net = layers.Dense(z_size * 2)

    def get_initial_state(self, batch_size):
        return (tf.zeros([batch_size, self.h_size]),
                tf.zeros([batch_size, self.z_size]))

    def observe(self, embed, action, state):
        h, z = state
        h, _ = self.cell(tf.concat([z, action], axis=-1), [h])
        post_out = self.post_net(tf.concat([h, embed], axis=-1))
        mean, std = tf.split(post_out, 2, axis=-1)
        std = tf.nn.softplus(std) + 0.1
        z = mean + std * tf.random.normal(tf.shape(mean))
        return (h, z), (mean, std)

    def imagine(self, action, state):
        h, z = state
        h, _ = self.cell(tf.concat([z, action], axis=-1), [h])
        prior_out = self.prior_net(h)
        mean, std = tf.split(prior_out, 2, axis=-1)
        std = tf.nn.softplus(std) + 0.1
        z = mean + std * tf.random.normal(tf.shape(mean))
        return (h, z), (mean, std)

# --- 2. Encoder, Decoder, Reward ---

def build_encoder(obs_shape, hidden_nodes=256):
    return tf.keras.Sequential([
        layers.InputLayer(shape=obs_shape),
        layers.Dense(hidden_nodes, activation='elu'),
        layers.Dense(hidden_nodes, activation='elu'),
    ])

def build_decoder(obs_shape, h_size, z_size, hidden_nodes=256):
    return tf.keras.Sequential([
        layers.InputLayer(shape=(h_size + z_size,)),
        layers.Dense(hidden_nodes, activation='elu'),
        layers.Dense(hidden_nodes, activation='elu'),
        layers.Dense(obs_shape[0])
    ])

def build_reward_model(h_size, z_size, hidden_nodes):
    return tf.keras.Sequential([
        layers.InputLayer(shape=(h_size + z_size,)),
        layers.Dense(hidden_nodes, activation='elu'),
        layers.Dense(1)
    ])

# --- 3. The Actor-Critic Agent ---

class Actor(tf.keras.Model):
    def __init__(self, action_dim, hidden_nodes=256):
        super().__init__()
        self.net = tf.keras.Sequential([
            layers.Dense(hidden_nodes, activation='elu'),
            layers.Dense(hidden_nodes, activation='elu'),
            layers.Dense(action_dim, activation='tanh')
        ])
    def call(self, latent):
        return self.net(latent)

class Critic(tf.keras.Model):
    def __init__(self, hidden_nodes=256):
        super().__init__()
        self.net = tf.keras.Sequential([
            layers.Dense(hidden_nodes, activation='elu'),
            layers.Dense(hidden_nodes, activation='elu'),
            layers.Dense(1)
        ])
    def call(self, latent):
        return self.net(latent)

class DreamerV1:
    def __init__(self, obs_shape, action_dim, 
                 h_size=256, 
                 z_size=32, 
                 batch_size=24, 
                 seq_len=50, 
                 horizon=15, 
                 kl_scale=0.1,
                 hidden_nodes=256,
                 lr_model=3e-4,
                 lr_agent=1e-4,
                 gamma=0.99,
                 lambda_=0.95,
                 actor_noise_init=0.3,
                 actor_noise_min=0.05,
                 actor_noise_decay=50000,
                 target_critic_tau=0.01,
                 model_grad_clip=100.0,
                 agent_grad_clip=100.0,
                 bufferlen=100000):
        self.obs_shape = obs_shape
        self.action_dim = action_dim
        self.rssm = RSSM(h_size, z_size)
        self.encoder = build_encoder(obs_shape, hidden_nodes)
        self.decoder = build_decoder(obs_shape, h_size, z_size, hidden_nodes)
        self.reward_model = build_reward_model(h_size, z_size, hidden_nodes)
        self.actor = Actor(action_dim, hidden_nodes)
        self.critic = Critic(hidden_nodes)
        self.critic_target = Critic(hidden_nodes)
        self.model_opt = tf.keras.optimizers.Adam(lr_model)
        self.agent_opt = tf.keras.optimizers.Adam(lr_agent)
        self.buffer = collections.deque(maxlen=bufferlen)
        self.HORIZON = horizon 
        self.SEQ_LEN = seq_len 
        self.BATCH_SIZE = batch_size
        self.H_SIZE = h_size 
        self.Z_SIZE = z_size
        self.KL_SCALE = kl_scale  
        self.GAMMA = gamma
        self.LAMBDA = lambda_
        self.ACTOR_NOISE_INIT = actor_noise_init
        self.ACTOR_NOISE_MIN = actor_noise_min
        self.ACTOR_NOISE_DECAY = float(actor_noise_decay)
        self.TARGET_CRITIC_TAU = target_critic_tau
        self.MODEL_GRAD_CLIP = model_grad_clip
        self.AGENT_GRAD_CLIP = agent_grad_clip
        self.train_step = 0

        # Build target critic with a dummy call and sync weights with the online critic.
        dummy_latent = tf.zeros([1, self.H_SIZE + self.Z_SIZE], dtype=tf.float32)
        _ = self.critic(dummy_latent)
        _ = self.critic_target(dummy_latent)
        self.critic_target.set_weights(self.critic.get_weights())


    def select_action(self, obs, state, prev_action=None, training=True):
        obs = tf.convert_to_tensor(obs[None], dtype=tf.float32)
        embed = self.encoder(obs)
        if prev_action is None:
            prev_action = tf.zeros([1, self.action_dim], dtype=tf.float32)
        else:
            prev_action = tf.convert_to_tensor(prev_action[None], dtype=tf.float32)
        state, _ = self.rssm.observe(embed, prev_action, state)
        h, z = state
        latent = tf.concat([h, z], axis=-1)
        action = self.actor(latent)
        if training:
            frac = np.exp(-float(self.train_step) / self.ACTOR_NOISE_DECAY)
            noise_std = self.ACTOR_NOISE_MIN + (self.ACTOR_NOISE_INIT - self.ACTOR_NOISE_MIN) * frac
            action += tf.random.normal(tf.shape(action)) * noise_std
            self.train_step += 1
        return tf.clip_by_value(action[0], -1.0, 1.0), state

    def train(self):
        if len(self.buffer) < self.BATCH_SIZE * self.SEQ_LEN:
            return None, None

        max_start = len(self.buffer) - self.SEQ_LEN
        if max_start <= 0:
            return None, None

        # Convert once for contiguous slicing; avoid scanning the whole replay every train call.
        replay = list(self.buffer)
        indices = []
        attempts = 0
        max_attempts = self.BATCH_SIZE * 40
        while len(indices) < self.BATCH_SIZE and attempts < max_attempts:
            idx = np.random.randint(0, max_start)
            seq = replay[idx:idx + self.SEQ_LEN]
            dones = [s[3] if len(s) > 3 else False for s in seq]
            if any(dones[:-1]):
                attempts += 1
                continue
            indices.append(idx)
            attempts += 1

        if len(indices) < self.BATCH_SIZE:
            return None, None

        obs_seq, act_seq, rew_seq = [], [], []
        for idx in indices:
            seq = replay[idx:idx+self.SEQ_LEN]
            obs_seq.append([s[0] for s in seq])
            act_seq.append([s[1] for s in seq])
            rew_seq.append([s[2] for s in seq])

        obs_seq = tf.convert_to_tensor(obs_seq, dtype=tf.float32)
        act_seq = tf.convert_to_tensor(act_seq, dtype=tf.float32)
        rew_seq = tf.convert_to_tensor(rew_seq, dtype=tf.float32)

        with tf.GradientTape() as model_tape:
            state = self.rssm.get_initial_state(self.BATCH_SIZE)
            enc_obs = self.encoder(tf.reshape(obs_seq, [-1, *self.obs_shape]))
            enc_obs = tf.reshape(enc_obs, [self.BATCH_SIZE, self.SEQ_LEN, -1])
            h_seq, z_seq = [], []
            kl_loss = 0.0
            for t in range(self.SEQ_LEN):
                h, z_prev = state
                h, _ = self.rssm.cell(tf.concat([z_prev, act_seq[:, t]], axis=-1), [h])
                prior_out = self.rssm.prior_net(h)
                p_m, p_s = tf.split(prior_out, 2, axis=-1)
                p_s = tf.nn.softplus(p_s) + 0.1
                post_out = self.rssm.post_net(tf.concat([h, enc_obs[:, t]], axis=-1))
                po_m, po_s = tf.split(post_out, 2, axis=-1)
                po_s = tf.nn.softplus(po_s) + 0.1
                z = po_m + po_s * tf.random.normal(tf.shape(po_m))
                state = (h, z)
                h_seq.append(h)
                z_seq.append(z)
                kl = tf.reduce_mean(tf.math.log(p_s/po_s) + (tf.square(po_s)+tf.square(po_m-p_m))/(2.0*tf.square(p_s)) - 0.5)
                kl_loss += tf.maximum(kl, 1.0)

            kl_loss = kl_loss / float(self.SEQ_LEN)

            latent_seq = tf.concat([tf.stack(h_seq, 1), tf.stack(z_seq, 1)], -1)
            rec_obs = self.decoder(tf.reshape(latent_seq, [-1, self.H_SIZE+self.Z_SIZE]))
            rec_loss = tf.reduce_mean(tf.square(rec_obs - tf.reshape(obs_seq, [-1, *self.obs_shape])))
            pred_rew = self.reward_model(tf.reshape(latent_seq, [-1, self.H_SIZE+self.Z_SIZE]))
            rew_loss = tf.reduce_mean(tf.square(tf.reshape(pred_rew, [self.BATCH_SIZE, self.SEQ_LEN]) - rew_seq))
            model_loss = rec_loss + rew_loss + self.KL_SCALE * kl_loss

        with tf.GradientTape(persistent=True) as agent_tape:
            h_stack = tf.stack(h_seq, 1)
            z_stack = tf.stack(z_seq, 1)
            h_flat = tf.reshape(h_stack, [-1, self.H_SIZE])
            z_flat = tf.reshape(z_stack, [-1, self.Z_SIZE])
            state = (tf.stop_gradient(h_flat), tf.stop_gradient(z_flat))
            imag_rewards, imag_values, target_values = [], [], []
            for _ in range(self.HORIZON):
                latent = tf.concat(state, -1)
                action = self.actor(latent)
                state, _ = self.rssm.imagine(action, state)
                next_latent = tf.concat(state, -1)
                imag_rewards.append(self.reward_model(next_latent))
                imag_values.append(self.critic(next_latent))
                target_values.append(self.critic_target(next_latent))

            imag_rewards = tf.reshape(tf.stack(imag_rewards), [self.HORIZON, -1])
            imag_values = tf.reshape(tf.stack(imag_values), [self.HORIZON, -1])
            target_values = tf.reshape(tf.stack(target_values), [self.HORIZON, -1])

            # Lambda-return target with a slowly-updated target critic for stability.
            targets = []
            ret = imag_rewards[-1] + self.GAMMA * target_values[-1]
            targets.append(ret)
            for t in reversed(range(self.HORIZON - 1)):
                ret = imag_rewards[t] + self.GAMMA * ((1.0 - self.LAMBDA) * target_values[t + 1] + self.LAMBDA * ret)
                targets.append(ret)
            returns = tf.stack(list(reversed(targets)), axis=0)

            # Actor optimizes the online value estimate on imagined states.
            actor_loss = -tf.reduce_mean(returns)
            critic_loss = tf.reduce_mean(tf.square(imag_values - tf.stop_gradient(returns)))
            agent_loss = actor_loss + critic_loss

        model_grads = model_tape.gradient(model_loss, self.model_opt_vars)
        model_grads, _ = tf.clip_by_global_norm(model_grads, self.MODEL_GRAD_CLIP)
        self.model_opt.apply_gradients(zip(model_grads, self.model_opt_vars))

        agent_grads = agent_tape.gradient(agent_loss, self.agent_opt_vars)
        agent_grads, _ = tf.clip_by_global_norm(agent_grads, self.AGENT_GRAD_CLIP)
        self.agent_opt.apply_gradients(zip(agent_grads, self.agent_opt_vars))

        # Soft-update target critic.
        tau = self.TARGET_CRITIC_TAU
        for target_var, source_var in zip(self.critic_target.trainable_variables, self.critic.trainable_variables):
            target_var.assign((1.0 - tau) * target_var + tau * source_var)
        del agent_tape
        return model_loss.numpy(), agent_loss.numpy()

    @property
    def model_opt_vars(self):
        return self.encoder.trainable_variables + self.decoder.trainable_variables + self.rssm.trainable_variables + self.reward_model.trainable_variables

    @property
    def agent_opt_vars(self):
        return self.actor.trainable_variables + self.critic.trainable_variables