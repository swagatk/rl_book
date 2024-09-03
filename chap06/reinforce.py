"""
Monte-Carlo Policy Gradient
REINFORCE algorithm
Credits: Sthanikam Santosh
URL: https://medium.com/@sthanikamsanthosh1994/reinforcement-learning-part-2-policy-gradient-reinforce-using-tensorflow2-a386a11e1dc6
"""

import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np

class PolicyNetwork():
    def __init__(self, obs_shape, n_actions, lr=0.0001, fc1_dim=256, fc2_dim=256):
        self.fc1_dim = fc1_dim
        self.fc2_dim = fc2_dim
        self.n_actions = n_actions
        self.obs_shape = obs_shape
        self.lr = lr
        self.model = self._build_model()
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr)
        
    def _build_model(self):
        inputs = tf.keras.layers.Input(shape=self.obs_shape)
        fc1 = tf.keras.layers.Dense(self.fc1_dim, activation='relu')(inputs)
        fc2 = tf.keras.layers.Dense(self.fc2_dim, activation='relu')(fc1)
        outputs = tf.keras.layers.Dense(self.n_actions, activation='softmax')(fc2)
        model = tf.keras.models.Model(inputs, outputs, name='policy_network')
        model.summary()
        return model
        
    def __call__(self, state):
        pi = self.model(state)
        return pi


class REINFORCEAgent:
    def __init__(self, obs_shape, n_actions, alpha=0.0005, gamma=0.99):
        self.gamma = gamma
        self.lr = alpha
        self.n_actions = n_actions
        self.states = []
        self.actions = []
        self.rewards = []
        self.obs_shape = obs_shape
        # create policy network
        self.policy = PolicyNetwork(obs_shape, n_actions, lr=self.lr)
            
    def choose_action(self, obs):
        state = tf.convert_to_tensor(obs, dtype=tf.float32)
        probs = self.policy(state)
        action_probs = tfp.distributions.Categorical(probs=probs)
        action = action_probs.sample()
        return action.numpy()[0]
    
    def store_transitions(self, state, action, reward):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        
    def calculate_return(self, rewards):
        G = np.zeros_like(rewards)
        for t in range(len(rewards)):
            G_sum = 0
            for k in range(t, len(rewards)):
                G_sum += rewards[k] * self.gamma
            G[t] = G_sum
        return G
    
    def train(self):
        actions = tf.convert_to_tensor(self.actions, dtype=tf.float32)
        rewards = np.array(self.rewards)
        # compute returns
        G = self.calculate_return(rewards)
        # optimize with gradient ascent
        with tf.GradientTape() as tape:
            loss = 0
            for idx, (g, state) in enumerate(zip(G, self.states)):
                state = tf.convert_to_tensor(state, dtype=tf.float32)
                probs = self.policy(state)
                action_probs = tfp.distributions.Categorical(probs=probs)
                log_prob = action_probs.log_prob(actions[idx])
                loss += -g * tf.squeeze(log_prob)
        trainable_params = self.policy.model.trainable_variables
        gradient = tape.gradient(loss,  trainable_params)
        self.policy.optimizer.apply_gradients(zip(gradient, trainable_params))
        # empty the buffer
        self.states = []
        self.actions = []
        self.rewards = []
        
        
    def save_weights(self, filename: str):
        if filename.lower().endswith(".weights.h5"):
            self.policy.model.save_weights(filename)
        else:
            raise ValueError("filename must end with '.weights.h5'")
        
    def load_weights(self, filename: str):
        if filename.lower().endswith(".weights.h5"):
            self.policy.model.load_weights(filename)
        else:
            raise ValueError("filename must end with '.weights.h5'")
        
    