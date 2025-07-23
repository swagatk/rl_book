"""
Deep Deterministic Policy Gradient (DDPG) Algorithm
"""
import tensorflow as tf
import numpy as np
from chap05.buffer import ReplayBuffer


# Noise Model
class OUActionNoise:
    def __init__(self, mean, std_deviation, theta=0.15, dt=1e-2, x_initial=None):
        self.theta = theta
        self.mean = mean
        self.std_dev = std_deviation
        self.dt = dt
        self.x_initial = x_initial
        self.reset()

    def __call__(self):
        # Formula taken from https://www.wikipedia.org/wiki/Ornstein-Uhlenbeck_process.
        x = (
            self.x_prev
            + self.theta * (self.mean - self.x_prev) * self.dt
            + self.std_dev * np.sqrt(self.dt) * np.random.normal(size=self.mean.shape)
        )
        # Store x into x_prev
        # Makes next noise dependent on current one
        self.x_prev = x
        return x

    def reset(self):
        if self.x_initial is not None:
            self.x_prev = self.x_initial
        else:
            self.x_prev = np.zeros_like(self.mean)


# Actor class
import tensorflow as tf
import numpy as np
class Actor():
    def __init__(self, obs_shape, action_size,
                 learning_rate=0.0003, 
                 action_upper_bound=1.0,
                model=None):
        self.obs_shape = obs_shape
        self.action_size = action_size
        self.lr = learning_rate
        self.action_upper_bound = action_upper_bound
        if model is None:
            self.model = self._build_model()
            self.target = self._build_model()
        else:
            self.model = model 
            self.target = tf.keras.models.clone_model(model)
        self.optimizer = tf.keras.optimizers.Adam()
        # target shares same weights as the primary model in the beginning
        self.target.set_weights(self.model.get_weights())
        
    def _build_model(self):
        initializer = tf.keras.initializers.RandomUniform(minval=-0.01, maxval=0.01)
        s_input = tf.keras.layers.Input(shape=self.obs_shape)
        fc1 = tf.keras.layers.Dense(256, activation='relu',
                                   kernel_initializer=initializer)(s_input)
        fc2 = tf.keras.layers.Dense(256, activation='relu',
                                   kernel_initializer=initializer)(fc1)
        a_out = tf.keras.layers.Dense(self.action_size, activation='tanh',
                                     kernel_initializer=initializer)(fc2)
        a_out = a_out * self.action_upper_bound
        model = tf.keras.models.Model(s_input, a_out, name='actor')
        model.summary()
        return model
    
    def __call__(self, states, target=False):
        if not target:
            pi = self.model(states)
        else:
            pi = self.target(states)
        return pi
    
    def update_target(self, tau=0.01):
        model_weights = self.model.get_weights()
        target_weights = self.target.get_weights()
        # Ensure shapes match
        if len(model_weights) != len(target_weights):
            raise ValueError('Model and Target should have same number of items')
        # update weights layer-by-layer using Polyak Averaging
        new_weights = []
        for w, w_dash in zip(model_weights, target_weights):
            new_w = tau * w + (1 - tau) * w_dash
            new_weights.append(new_w)
        self.target.set_weights(new_weights)
        
    def train(self, states, critic):
        with tf.GradientTape() as tape:
            actor_weights = self.model.trainable_variables
            actions = self.model(states)
            critic_values = critic(states, actions)
            # -ve value is used to maximize the function
            actor_loss = -tf.math.reduce_mean(critic_values)
        actor_grad = tape.gradient(actor_loss, actor_weights)
        self.optimizer.apply_gradients(zip(actor_grad, actor_weights))
        return actor_loss
    
    def save_weights(self, filename: str):
        if filename.lower().endswith(".weights.h5"):
            self.model.save_weights(filename)
        else:
            raise ValueError("filename must end with '.weights.h5'")
        
    def load_weights(self, filename: str):
        if filename.lower().endswith(".weights.h5"):
            self.model.load_weights(filename)
        else:
            raise ValueError("filename must end with '.weights.h5'")


# Critic Class
class Critic:
    def __init__(self, obs_shape, action_size,
                learning_rate=0.0003,
                gamma=0.99,
                model=None):
        self.obs_shape = obs_shape
        self.action_size = action_size
        self.gamma = gamma
        self.lr = learning_rate
        
        if model is None:
            self.model = self._build_model()
            self.target = self._build_model()
        else:
            self.model = model
            self.target = tf.keras.models.clone_model(model)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr)
        
        # target shares same weights as the main model in the beginning
        self.target.set_weights(self.model.get_weights())
        
    def _build_model(self):
        s_input = tf.keras.layers.Input(shape=self.obs_shape)
        s_out = tf.keras.layers.Dense(16, activation='relu')(s_input)
        s_out = tf.keras.layers.Dense(32, activation='relu')(s_out)
        
        # action as input
        a_input = tf.keras.layers.Input(shape=(action_size, ))
        a_out = tf.keras.layers.Dense(32, activation='relu')(a_input)
        
        # concat [s, a]
        concat = tf.keras.layers.Concatenate()([s_out, a_out])
        out = tf.keras.layers.Dense(256, activation='relu')(concat)
        out = tf.keras.layers.Dense(256, activation='relu')(out)
        net_out = tf.keras.layers.Dense(1)(out)
        
        # output is the Q-value output
        model = tf.keras.models.Model(inputs=[s_input, a_input], 
                                      outputs=net_out, name='critic')
        model.summary()
        return model
    
    def __call__(self, states, actions, target=False):
        if not target:
            value = self.model([states, actions])
        else:
            value = self.target([states, actions])
        return value
        
    def update_target(self, tau=0.01):
        model_weights = self.model.get_weights()
        target_weights = self.target.get_weights()
        # Ensure shapes match
        if len(model_weights) != len(target_weights):
            raise ValueError('Model and Target should have same number of items')
        # update weights layer-by-layer using Polyak Averaging
        new_weights = []
        for w, w_dash in zip(model_weights, target_weights):
            new_w = tau * w + (1 - tau) * w_dash
            new_weights.append(new_w)
        self.target.set_weights(new_weights)
        
    def train(self, states, actions, rewards, next_states, dones, actor):
        with tf.GradientTape() as tape:
            critic_weights = self.model.trainable_variables
            target_actions = actor(states, target=True)
            target_q_values = self.target([next_states, target_actions])
            y = rewards + self.gamma * (1 - dones) * target_q_values
            q_values = self.model([states, actions])
            critic_loss = tf.math.reduce_mean(tf.square(y - q_values))
        critic_grads = tape.gradient(critic_loss, critic_weights)
        self.optimizer.apply_gradients(zip(critic_grads, critic_weights))
        return critic_loss
    
    def save_weights(self, filename: str):
        if filename.lower().endswith(".weights.h5"):
            self.model.save_weights(filename)
        else:
            raise ValueError("filename must end with '.weights.h5'")
        
    def load_weights(self, filename: str):
        if filename.lower().endswith(".weights.h5"):
            self.model.load_weights(filename)
        else:
            raise ValueError("filename must end with '.weights.h5'")


# DDPG Agent
class DDPGAgent:
    def __init__(self, obs_shape, action_size,
                 batch_size, buffer_capacity,
                 action_upper_bound=1.0,
                 action_lower_bound=-1.0,
                lr_a=1e-3, lr_c=1e-3, gamma=0.99,
                 noise_std=0.2,
                actor_model=None,
                critic_model=None):
        self.obs_shape = obs_shape
        self.action_size = action_size
        self.buffer_capacity = buffer_capacity
        self.batch_size = batch_size
        self.action_upper_bound = action_upper_bound
        self.action_lower_bound = action_lower_bound
        self.gamma = gamma
        self.lr_a = lr_a
        self.lr_c = lr_c
        self.noise_std = noise_std
        
        self.actor = Actor(self.obs_shape, self.action_size, 
                           learning_rate=self.lr_a,
                          action_upper_bound=self.action_upper_bound,
                          model=actor_model)
        self.critic = Critic(self.obs_shape, self.action_size,
                            learning_rate=self.lr_c,
                            gamma=self.gamma,
                            model=critic_model)
        self.buffer = ReplayBuffer(self.buffer_capacity)
        self.action_noise = OUActionNoise(mean=np.zeros(1), 
                            std_deviation=float(self.noise_std) * np.ones(1))
        
    def policy(self, state):
        action = tf.squeeze(self.actor(state))
        noise = self.action_noise()
        # add noise to action
        sampled_action = action.numpy() + noise
        # check action bounds
        valid_action = np.clip(sampled_action, self.action_lower_bound, 
                                           self.action_upper_bound)
        return valid_action
        
    def experience_replay(self):
        if len(self.buffer) < self.batch_size:
            return
        mini_batch = self.buffer.sample(self.batch_size)
        states = np.zeros((self.batch_size, *self.obs_shape))
        next_states = np.zeros((self.batch_size, *self.obs_shape))
        actions = np.zeros((self.batch_size, self.action_size))
        rewards = np.zeros((self.batch_size, 1))
        dones = np.zeros((self.batch_size, 1))
        for i in range(len(mini_batch)):
            states[i] = mini_batch[i][0]
            actions[i] = mini_batch[i][1]
            rewards[i] = mini_batch[i][2]
            next_states[i] = mini_batch[i][3]
            dones[i] = mini_batch[i][4]
        states = tf.convert_to_tensor(states, dtype=tf.float32)
        actions = tf.convert_to_tensor(actions, dtype=tf.float32)
        rewards = tf.convert_to_tensor(rewards, dtype=tf.float32)
        next_states = tf.convert_to_tensor(next_states, dtype=tf.float32)
        dones = tf.convert_to_tensor(dones, dtype=tf.float32)
        # train
        a_loss = self.actor.train(states, self.critic)
        c_loss = self.critic.train(states, actions, rewards, next_states, dones, self.actor)
        return a_loss, c_loss
    
    def update_targets(self, tau_a=0.01, tau_c=0.02):
        self.actor.update_target(tau_a)
        self.critic.update_target(tau_c)
        
    def save_weights(self, actor_wt_file='actor.weights.h5', 
                    critic_wt_file='critic.weights.h5'):
        self.actor.save_weights(actor_wt_file)
        self.critic.save_weights(critic_wt_file)
        
    def load_weights(self, actor_wt_file='actor.weights.h5', 
                     critic_wt_file='critic.weights.h5'):
        self.actor.load_weights(actor_wt_file)
        self.critic.load_weights(critic_wt_file)

