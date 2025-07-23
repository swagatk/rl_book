"""
SAC Algorithm Implementation using TensorFlow 2.x
- It uses two critics to mitigate overestimation bias
"""
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

        if model is None:
            self.model = self._build_model()
        else:
            self.model = tf.keras.models.clone_model(model)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr)

        # Constraints for log_std to ensure numerical stability and reasonable exploration
        # log_std values are usually clipped to a range like [-20, 2] or [-5, 2]        
        self.max_log_std = tf.constant(2.0, dtype=tf.float32)
        self.min_log_std = tf.constant(-20.0, dtype=tf.float32)
        
    def _build_model(self):
        inp = tf.keras.layers.Input(shape=self.obs_shape)
        f = tf.keras.layers.Dense(256, activation='relu')(inp)
        f = tf.keras.layers.Dense(256, activation='relu')(f)
        mu = tf.keras.layers.Dense(self.action_size, activation=None)(f)
        log_std = tf.keras.layers.Dense(self.action_size, activation=None)(f)
        model = tf.keras.models.Model(inputs=inp, outputs=[mu, log_std], name='actor')
        model.summary()
        return model
    
    def __call__(self, states):
        mean, log_std = self.model(states)
        log_std = tf.clip_by_value(log_std, self.min_log_std, self.max_log_std)
        return mean, log_std

    def sample_action(self, state, reparameterize=False):
        mean, log_std = self(state)
        std = tf.exp(log_std)
        dist = tfp.distributions.Normal(mean, std)

        if reparameterize: # Reparameterization trick
            z = mean + std * tf.random.normal(tf.shape(mean))
        else:
            z = dist.sample()

        action = tf.tanh(z) * self.max_action 
        log_prob = dist.log_prob(z) 
        log_prob -= tf.math.log(1 - tf.math.pow(action, 2) + 1e-6)
        log_prob = tf.math.reduce_sum(log_prob,  axis=-1, keepdims=True)
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
        return self.model([states, actions])
    
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
    def __init__(self, obs_shape, action_shape, 
                 actor_lr=0.0003, critic_lr=0.0003, 
                 alpha_lr=1e-4, alpha=0.1,
                 gamma=0.99, polyak=0.995, 
                 action_upper_bound=1.0,
                 buffer_size=100000, 
                 batch_size=256,
                 reward_scale=1.0,
                 max_grad_norm=None, # required for gradient clipping
                 actor_model=None,
                 critic_model=None ):

        self.obs_shape = obs_shape
        self.action_shape = action_shape
        self.gamma = gamma  # Discount factor
        self.polyak = polyak # Polyak averaging coefficient
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.alpha_lr = alpha_lr
        self.reward_scale = reward_scale
        self.max_grad_norm = max_grad_norm
        self.name = 'SAC'

        # Initialize actor and critic networks
        self.actor = Actor(obs_shape, action_shape, self.actor_lr, action_upper_bound, model=actor_model)
        self.critic_1 = Critic(obs_shape, action_shape, self.critic_lr, model=critic_model)
        self.critic_2 = Critic(obs_shape, action_shape, self.critic_lr, model=critic_model)

        # Target networks for soft updates
        self.target_critic_1 = Critic(obs_shape, action_shape, self.critic_lr, model=critic_model)
        self.target_critic_2 = Critic(obs_shape, action_shape, self.critic_lr, model=critic_model)

        # make alpha a trainable variable
        self.log_alpha = tf.Variable(tf.math.log(alpha), dtype=tf.float32, trainable=True, name='log_alpha')
        self.alpha_optimizer = tf.keras.optimizers.Adam(learning_rate=self.alpha_lr)

        # Target entropy for action space 
        self.target_entropy = tf.constant(-np.prod(self.action_shape), dtype=tf.float32)  

        # Replay buffer
        self.buffer = ReplayBuffer(self.buffer_size)  

        # Initialize target networks with the same weights as the main networks
        self.target_critic_1.model.set_weights(self.critic_1.model.get_weights())
        self.target_critic_2.model.set_weights(self.critic_2.model.get_weights())

    def update_target_networks(self):
        """
        Soft update target networks using Polyak averaging
        """
        for target_var, var in zip(self.target_critic_1.model.trainable_variables, 
                                self.critic_1.model.trainable_variables):
            target_var.assign(self.polyak * target_var + (1 - self.polyak) * var)

        for target_var, var in zip(self.target_critic_2.model.trainable_variables, 
                                self.critic_2.model.trainable_variables):
            target_var.assign(self.polyak * target_var + (1 - self.polyak) * var)

    def choose_action(self, state, evaluate=False):
        """
        Choose an action based on the current state
        """
        state = tf.expand_dims(tf.convert_to_tensor(state, dtype=tf.float32), axis=0)
        action, _ = self.actor.sample_action(state, reparameterize=not evaluate)
        action = tf.squeeze(action, axis=0)  # Remove batch dimension
        return action.numpy()

    def store_transition(self, state, action, reward, next_state, done):
        """
        Store a transition in the replay buffer
        : inputs are numpy arrays
        """
        self.buffer.add((state, action, reward, next_state, done))

    def update_critic(self, states, actions, rewards, next_states, dones):
        """
        Update the critic networks using the sampled transitions
        : inputs are tensors
        """

        with tf.GradientTape(persistent=True) as tape:
            # predict next_actions and log_probs for next states
            next_actions, next_log_probs = self.actor.sample_action(next_states)

            # compute target Q-values using the target critic networks
            target_q1 = self.target_critic_1(next_states, next_actions)
            target_q2 = self.target_critic_2(next_states, next_actions)
            min_target_q_next = tf.minimum(target_q1, target_q2) 

            # Soft Bellman backup equation for target Q-value
            # y = r + gamma * (1 - done) * (min_Q_target(s', a') - alpha * log_pi(a'|s'))
            target_q_values = self.reward_scale * rewards + self.gamma * (tf.ones_like(dones) - dones) * \
                        (min_target_q_next - tf.exp(self.log_alpha) * next_log_probs)
            
            # compute current Q-values
            current_q1 = self.critic_1(states, actions)
            current_q2 = self.critic_2(states, actions)

            # compute critic losses
            critic_1_loss = tf.reduce_mean(tf.square(current_q1 - target_q_values))
            critic_2_loss = tf.reduce_mean(tf.square(current_q2 - target_q_values))

        # compute gradients for critic networks
        critic_1_grads = tape.gradient(critic_1_loss, self.critic_1.model.trainable_variables)
        critic_2_grads = tape.gradient(critic_2_loss, self.critic_2.model.trainable_variables)  

        if self.max_grad_norm is not None:
            # Clip gradients to avoid exploding gradients
            critic_1_grads, _ = tf.clip_by_global_norm(critic_1_grads, self.max_grad_norm)
            critic_2_grads, _ = tf.clip_by_global_norm(critic_2_grads, self.max_grad_norm)

        # apply gradients to the critic networks if gradients are not None
        if critic_1_grads is not None:
            self.critic_1.optimizer.apply_gradients(zip(critic_1_grads, 
                                            self.critic_1.model.trainable_variables))
        if critic_2_grads is not None:
            self.critic_2.optimizer.apply_gradients(zip(critic_2_grads, 
                                            self.critic_2.model.trainable_variables))

        mean_c_loss = (critic_1_loss + critic_2_loss) / 2.0
        return mean_c_loss

    def update_actor(self, states, actions):
        """
        Update the actor network
        inputs are tensors
        outputs: actor_loss and alpha_loss
        """

        with tf.GradientTape(persistent=True) as tape:
            # Sample actions and log probabilities for current states
            actions, log_probs = self.actor.sample_action(states)

            # Compute Q-values for the sampled actions
            q1 = self.critic_1(states, actions)
            q2 = self.critic_2(states, actions)
            min_q = tf.minimum(q1, q2)

            # Actor loss is the mean of the Q-values minus the entropy term
            # Actor loss (maximize soft Q-value, incorporating entropy)
            # J_pi = E_s,a~pi [alpha * log_pi(a|s) - Q(s,a)] -> minimize -J_pi
            actor_loss = tf.reduce_mean(tf.exp(self.log_alpha) * log_probs - min_q)

            # alpha loss is computed as the mean of the target entropy minus the log probability
            alpha_loss = tf.reduce_mean(tf.negative(self.log_alpha) * (log_probs + self.target_entropy))

        # Compute gradients for the actor network
        actor_grads = tape.gradient(actor_loss, self.actor.model.trainable_variables)
        # Compute gradients for alpha
        alpha_grads = tape.gradient(alpha_loss, [self.log_alpha])

        if self.max_grad_norm is not None:
            # Clip gradients to avoid exploding gradients
            actor_grads, _ = tf.clip_by_global_norm(actor_grads, self.max_grad_norm)
            alpha_grads, _ = tf.clip_by_global_norm(alpha_grads, self.max_grad_norm)

        # Apply gradients to the actor network if gradients are not None
        if actor_grads is not None:
            self.actor.optimizer.apply_gradients(zip(actor_grads, 
                                            self.actor.model.trainable_variables)) 

        # Apply gradients to alpha if gradients are not None
        if alpha_grads is not None: 
            self.alpha_optimizer.apply_gradients(zip(alpha_grads, [self.log_alpha]))
        
        return actor_loss, alpha_loss

    #@tf.function
    def train(self, update_per_step=1):
        if len(self.buffer) < self.batch_size:
            return 0, 0, 0

        c_losses, a_losses, alpha_losses = [], [], []
        for _ in range(update_per_step):

            # Sample a batch of transitions from the replay buffer
            states, actions, rewards, next_states, dones = self.buffer.sample_unpacked(self.obs_shape, 
                                                                                       self.action_shape,
                                                                                       self.batch_size)

            states = tf.convert_to_tensor(states, dtype=tf.float32)
            actions = tf.convert_to_tensor(actions, dtype=tf.float32)       
            rewards = tf.convert_to_tensor(rewards, dtype=tf.float32)
            next_states = tf.convert_to_tensor(next_states, dtype=tf.float32)
            dones = tf.convert_to_tensor(dones, dtype=tf.float32)

            # Update Critic networks
            critic_loss = self.update_critic(states, actions, rewards, next_states, dones)
            actor_loss, alpha_loss = self.update_actor(states, actions)

            # Update target networks
            self.update_target_networks()

            c_losses.append(critic_loss)
            a_losses.append(actor_loss)
            alpha_losses.append(alpha_loss)

        mean_critic_loss = tf.reduce_mean(c_losses)
        mean_actor_loss = tf.reduce_mean(a_losses)
        mean_alpha_loss = tf.reduce_mean(alpha_losses)

        return mean_critic_loss, mean_actor_loss, mean_alpha_loss

    def save_weights(self, filename: str):
        if filename.lower().endswith(".weights.h5"):
            self.actor.save_weights(filename.replace('.weights.h5', '_actor.weights.h5'))
            self.critic_1.save_weights(filename.replace('.weights.h5', '_critic1.weights.h5'))
            self.critic_2.save_weights(filename.replace('.weights.h5', '_critic2.weights.h5'))
        else:
            raise ValueError("filename must end with '.weights.h5'")

    def load_weights(self, filename: str):
        if filename.lower().endswith(".weights.h5"):
            self.actor.load_weights(filename.replace('.weights.h5', '_actor.weights.h5'))
            self.critic_1.load_weights(filename.replace('.weights.h5', '_critic1.weights.h5'))
            self.critic_2.load_weights(filename.replace('.weights.h5', '_critic2.weights.h5'))
        else:
            raise ValueError("filename must end with '.weights.h5'")
        

