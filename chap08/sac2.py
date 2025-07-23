import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from collections import deque
import sys

sys.path.append("/Share/rl_book/chap05")  # Adjust path as needed
from buffer import ReplayBuffer

# Actor class
import tensorflow as tf
import numpy as np

class Actor():
    def __init__(self, obs_shape, action_shape,
                 learning_rate=0.0001, 
                 action_upper_bound=1.0,
                model=None):
        self.obs_shape = obs_shape
        self.action_size = action_shape[0]
        self.lr = learning_rate
        self.max_action = action_upper_bound
        self.noise = 1e-6  # Small value to avoid division by zero

        if model is None:
            self.model = self._build_model()
        else:
            self.model = tf.keras.models.clone_model(model)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr)

        
    def _build_model(self):
        inp = tf.keras.layers.Input(shape=self.obs_shape)
        f = tf.keras.layers.Dense(256, activation='relu')(inp)
        f = tf.keras.layers.Dense(256, activation='relu')(f)
        mu = tf.keras.layers.Dense(self.action_size, activation=None)(f)
        sigma = tf.keras.layers.Dense(self.action_size, activation=None)(f)
        model = tf.keras.models.Model(inputs=inp, outputs=[mu, sigma], name='actor')
        model.summary()
        return model
    
    def __call__(self, states):
        mean, std = self.model(states)
        # Apply constraints to log_std
        std = tf.clip_by_value(std, self.noise, 1.0)
        return mean, std

    def sample_action(self, state, reparameterize=False):
        mean, std = self(state)
        dist = tfp.distributions.Normal(mean, std)

        if reparameterize: # Reparameterization trick
            z = mean + std * tf.random.normal(tf.shape(mean))
        else:
            z = dist.sample()

        action = tf.tanh(z) * self.max_action  # Scale action to the range [-max_action, max_action]
        log_prob = dist.log_prob(z) 
        log_prob -= tf.math.log(1 - tf.math.pow(action, 2) + self.noise)
        log_prob = tf.math.reduce_sum(log_prob, axis=1, keepdims=True)
        return action, log_prob
    
    
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


class Critic:
    """
    Approximates Q(s,a) function
    """
    def __init__(self, obs_shape, action_shape, 
                 learning_rate=1e-3,
                 model=None):
        self.obs_shape = obs_shape    # shape: (m,)
        self.action_shape = action_shape  # shape: (n, )
        self.lr = learning_rate        
        # create NN model
        if model is None:
            self.model = self._build_net()
        else:
            self.model = tf.keras.models.clone_model(model)
        self.optimizer = tf.keras.optimizers.Adam(self.lr)

    def _build_net(self):
        state_input = tf.keras.layers.Input(shape=self.obs_shape)
        action_input = tf.keras.layers.Input(shape=self.action_shape)
        concat = tf.keras.layers.Concatenate()([state_input, action_input])
        out = tf.keras.layers.Dense(256, activation='relu')(concat)
        out = tf.keras.layers.Dense(256, activation='relu')(out)
        out = tf.keras.layers.Dense(256, activation='relu')(out)
        net_out = tf.keras.layers.Dense(1)(out)
        model = tf.keras.Model(inputs=[state_input, action_input], outputs=net_out,
                               name='critic')
        model.summary()
        return model

    def __call__(self, states, actions):
        """
        Returns Q(s,a) value
        """
        q = self.model([states, actions])
        return q

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


class ValueNetwork():
    """
    Approximates V(s) function
    """
    def __init__(self, obs_shape, 
                 learning_rate=1e-3,
                 model=None):
        self.obs_shape = obs_shape
        self.lr = learning_rate
        # create NN model
        if model is None:
            self.model = self._build_net()
        else:
            self.model = tf.keras.models.clone_model(model)
        self.optimizer = tf.keras.optimizers.Adam(self.lr)

    def _build_net(self):
        inp = tf.keras.layers.Input(shape=self.obs_shape)
        out = tf.keras.layers.Dense(256, activation='relu')(inp)
        out = tf.keras.layers.Dense(256, activation='relu')(out)
        out = tf.keras.layers.Dense(256, activation='relu')(out)
        net_out = tf.keras.layers.Dense(1)(out)
        model = tf.keras.Model(inputs=inp, outputs=net_out, name='value_network')
        model.summary()
        return model

    def __call__(self, states):
        """
        Returns V(s) value
        """
        v = self.model(states)  
        return v  
    
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


class SACAgent:
    """
    Soft Actor-Critic Agent
    """
    def __init__(self, obs_shape, action_shape,
                 action_upper_bound=1.0,
                 reward_scale=1.0,
                 buffer_size=100000,
                 batch_size=256,
                 lr_a=1e-3,
                 lr_c=1e-4,
                 lr_alpha=1e-4,
                 polyak = 0.995,
                 gamma = 0.99,
                 alpha=0.2,
                 max_grad_norm=None,
                 actor_model=None,
                 critic_model=None,
                 value_model=None):
        self.obs_shape = obs_shape
        self.action_shape = action_shape
        self.action_size = action_shape[0]
        self.action_upper_bound = action_upper_bound
        self.reward_scale = reward_scale
        self.batch_size = batch_size
        self.polyak = polyak
        self.gamma = gamma
        self.name = 'SAC2'
        self.target_entropy = -np.prod(action_shape)  # Target entropy = -|A|
        self.actor_lr = lr_a
        self.critic_lr = lr_c
        self.alpha_lr = lr_alpha
        self.buffer_size = buffer_size
        self.max_grad_norm = max_grad_norm
        

        # Initialize networks
        self.actor = Actor(obs_shape, action_shape, lr_a, action_upper_bound, model=actor_model)
        self.critic_1 = Critic(obs_shape, action_shape, learning_rate=self.actor_lr, model=critic_model)
        self.critic_2 = Critic(obs_shape, action_shape, learning_rate=self.critic_lr, model=critic_model)
        self.value_network = ValueNetwork(obs_shape, learning_rate=self.critic_lr, model = value_model)

        # create a replay buffer
        self.buffer = ReplayBuffer(self.buffer_size)

        # Target networks for stability
        self.target_value_network = ValueNetwork(obs_shape, learning_rate=self.critic_lr)
        self.target_value_network.model.set_weights(self.value_network.model.get_weights())

        # Initialize alpha (temperature parameter)
        self.alpha = tf.Variable(alpha, trainable=True, dtype=tf.float32)
        self.alpha_optimizer = tf.keras.optimizers.Adam(learning_rate=self.alpha_lr)


    def choose_action(self, state, reparameterize=False):
        """
        Chooses an action based on the current state
        """
        state = tf.expand_dims(tf.convert_to_tensor(state, dtype=tf.float32), axis=0)   
        action, _ = self.actor.sample_action(state, reparameterize)
        action = tf.squeeze(action, axis=0)  # Remove batch dimension
        return action.numpy()
    
    def store_transition(self, state, action, reward, next_state, done):
        """
        Stores a transition in the replay buffer
        """
        self.buffer.add((state, action, reward, next_state, done))

    def update_target_networks(self):
        """
        Update target networks using Polyak averaging
        """
        for target_var, var in zip(self.target_value_network.model.trainable_variables,
                                   self.value_network.model.trainable_variables):
            target_var.assign(self.polyak * target_var + (1 - self.polyak) * var)
            
    def update_value_network(self, states, next_states):
        with tf.GradientTape() as tape:
            next_actions, next_log_probs = self.actor.sample_action(next_states)
            next_q1 = self.critic_1(next_states, next_actions)
            next_q2 = self.critic_2(next_states, next_actions)
            next_q = tf.minimum(next_q1, next_q2)
            value_target = next_q - self.alpha * next_log_probs
            value = self.value_network(states)
            value_loss = tf.reduce_mean(tf.square(value - value_target))

        value_grads = tape.gradient(value_loss, self.value_network.model.trainable_variables)

        if self.max_grad_norm is not None:
            value_grads, _ = tf.clip_by_global_norm(value_grads, self.max_grad_norm)

        if value_grads is not None:
            # Apply gradients to the value network
            self.value_network.optimizer.apply_gradients(zip(value_grads, \
                                            self.value_network.model.trainable_variables))
        return value_loss.numpy() 

    def update_critic(self, states, actions, rewards, next_states, dones):
        with tf.GradientTape(persistent=True) as tape:
            q1 = self.critic_1(states, actions)
            q2 = self.critic_2(states, actions)
            next_v = self.target_value_network(next_states)
            target_q = self.reward_scale * rewards + self.gamma * (1 - dones) * next_v
            critic_1_loss = tf.reduce_mean(tf.square(target_q - q1))
            critic_2_loss = tf.reduce_mean(tf.square(target_q - q2))

        critic_1_grads = tape.gradient(critic_1_loss, self.critic_1.model.trainable_variables)
        critic_2_grads = tape.gradient(critic_2_loss, self.critic_2.model.trainable_variables)

        if self.max_grad_norm is not None:
            critic_1_grads, _ = tf.clip_by_global_norm(critic_1_grads, self.max_grad_norm)
            critic_2_grads, _ = tf.clip_by_global_norm(critic_2_grads, self.max_grad_norm)

        if critic_1_grads is not None:
            self.critic_1.optimizer.apply_gradients(zip(critic_1_grads, \
                                            self.critic_1.model.trainable_variables))
        if critic_2_grads is not None:
            self.critic_2.optimizer.apply_gradients(zip(critic_2_grads, \
                                            self.critic_2.model.trainable_variables))
        return critic_1_loss.numpy(), critic_2_loss.numpy()

    def update_actor(self, states):
        with tf.GradientTape(persistent=True) as tape:
            new_actions, log_probs = self.actor.sample_action(states)
            q1_new = self.critic_1(states, new_actions)
            q2_new = self.critic_2(states, new_actions)
            q_new = tf.minimum(q1_new, q2_new)
            actor_loss = tf.reduce_mean(self.alpha * log_probs - q_new)
            alpha_loss = tf.reduce_mean(-tf.math.log(self.alpha) * (log_probs + self.target_entropy))    

        actor_grads = tape.gradient(actor_loss, self.actor.model.trainable_variables)
        alpha_grads = tape.gradient(alpha_loss, [self.alpha])

        if self.max_grad_norm is not None:
            actor_grads, _ = tf.clip_by_global_norm(actor_grads, self.max_grad_norm)
            alpha_grads, _ = tf.clip_by_global_norm(alpha_grads, self.max_grad_norm)

        if actor_grads is not None:
            self.actor.optimizer.apply_gradients(zip(actor_grads, \
                                            self.actor.model.trainable_variables))
        if alpha_grads is not None:
            self.alpha_optimizer.apply_gradients(zip(alpha_grads, [self.alpha]))
        return actor_loss.numpy(), alpha_loss.numpy()

    @tf.function
    def train(self):
        """
        Train the agent using a batch of transitions from the replay buffer
        """
        if len(self.buffer) < self.batch_size:
            return 0, 0, 0, 0 
        states, actions, rewards, next_states, dones = self.buffer.sample_unpacked(self.obs_shape,
                                                                                   self.action_shape,
                                                                                   self.batch_size)

        # Convert to tensors
        states = tf.convert_to_tensor(states, dtype=tf.float32)
        actions = tf.convert_to_tensor(actions, dtype=tf.float32)
        rewards = tf.convert_to_tensor(rewards, dtype=tf.float32)
        next_states = tf.convert_to_tensor(next_states, dtype=tf.float32)
        dones = tf.convert_to_tensor(dones, dtype=tf.float32)

        # Update value network
        value_loss = self.update_value_network(states, next_states)

        # Update critic networks
        critic_1_loss, critic_2_loss = self.update_critic(states, actions, rewards, next_states, dones)

        critic_loss = np.mean([critic_1_loss, critic_2_loss]) 

        # Update actor network
        actor_loss, alpha_loss = self.update_actor(states)

        # Update target networks
        self.update_target_networks()
        return value_loss, critic_loss, actor_loss, alpha_loss

    def save_weights(self, filename: str):
        if filename.lower().endswith(".weights.h5"):
            self.actor.save_weights(filename)
            self.critic_1.save_weights(filename.replace('.weights.h5', '_critic1.weights.h5'))
            self.critic_2.save_weights(filename.replace('.weights.h5', '_critic2.weights.h5'))
            self.value_network.save_weights(filename.replace('.weights.h5', '_value.weights.h5'))
        else:
            raise ValueError("filename must end with '.weights.h5'")

    def load_weights(self, filename: str):
        if filename.lower().endswith(".weights.h5"):
            self.actor.load_weights(filename)
            self.critic_1.load_weights(filename.replace('.weights.h5', '_critic1.weights.h5'))
            self.critic_2.load_weights(filename.replace('.weights.h5', '_critic2.weights.h5'))
            self.value_network.load_weights(filename.replace('.weights.h5', '_value.weights.h5'))
        else:
            raise ValueError("filename must end with '.weights.h5'")