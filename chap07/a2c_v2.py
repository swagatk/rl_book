'''
A2C implementation with a single worker.

- It computes advantages using rewards, values, next_values, and dones. 
- It does not require computing return values for each episode. 
- TD error is used to estimate advantages.
'''

import tensorflow as tf
import numpy as np
import tensorflow_probability as tfp


#########################

######################
class Actor():
    def __init__(self, obs_shape, action_size, lr=1e-4, model=None):
        self.obs_shape = obs_shape
        self.action_size = action_size
        self.lr = lr
        if model is None:
            self.model = self._build_model()
        else:
            self.model = model 
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr)


    def _build_model(self): # outputs action probabilities
        sinput = tf.keras.layers.Input(shape=self.obs_shape)
        x = tf.keras.layers.Dense(512, activation='relu',
                                 kernel_initializer=tf.keras.initializers.HeUniform())(sinput)
        x = tf.keras.layers.Dense(512, activation='relu',
                                 kernel_initializer=tf.keras.initializers.HeUniform())(x)
        aout = tf.keras.layers.Dense(self.action_size, activation='softmax',
                                    kernel_initializer=tf.keras.initializers.HeUniform())(x)
        model = tf.keras.models.Model(sinput, aout, name='actor')
        model.summary()
        return model

    def __call__(self, states):
        # returns action probabilities for each state
        pi = tf.squeeze(self.model(states))
        return pi
    


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
        

###################
class Critic():
    def __init__(self, obs_shape,
                lr = 1e-4, gamma=0.99, model=None):
        self.obs_shape = obs_shape
        self.gamma = gamma
        self.lr = lr
        if model is None:
            self.model = self._build_model()
        else:
            self.model = model
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr)

    def _build_model(self): # returns V(s)
        sinput = tf.keras.layers.Input(shape=self.obs_shape)
        x = tf.keras.layers.Dense(128, activation='relu')(sinput)
        x = tf.keras.layers.Dense(256, activation='relu')(x)
        x = tf.keras.layers.Dense(256, activation='relu')(x)
        vout = tf.keras.layers.Dense(1, activation='relu')(x) # scalar output
        model = tf.keras.models.Model(inputs= sinput, outputs=vout, 
                                      name='critic')
        model.summary()
        return model

    def __call__(self, states):
        # returns V(s) for each state
        value = tf.squeeze(self.model(states))
        return value
    
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

##############
class A2CAgent:
    def __init__(self, obs_shape, action_size, 
                 lr_a=1e-4, lr_c=1e-3, gamma=0.99,
                 entropy_beta=0.01, grad_clip_norm=5.0,
                 a_model=None, c_model=None):
        self.lr_a = lr_a
        self.lr_c = lr_c
        self.entropy_beta = entropy_beta
        self.grad_clip_norm = grad_clip_norm
        
        self.gamma = gamma
        self.action_size = action_size
        self.obs_shape = obs_shape
        self.name = 'A2C_v2'

        # create actor and critic networks
        self.actor = Actor(obs_shape, action_size, lr=self.lr_a,
                           model=a_model)
        self.critic = Critic(obs_shape, lr=self.lr_c,
                             gamma=self.gamma, model=c_model)

    def policy(self, state):
        state = tf.expand_dims(tf.convert_to_tensor(state, dtype=tf.float32), axis=0)
        pi = self.actor(state)
        pi_np = pi.numpy()
        if np.isnan(pi_np).any() or np.isinf(pi_np).any():
            raise ValueError(f"NaN/inf detected in action probabilities: {pi_np}.")
        action_probs = tfp.distributions.Categorical(probs=pi)
        action = action_probs.sample()
        #action = tf.clip_by_value(action, 0, self.action_size - 1)  # ensure valid action
        return action.numpy()
    
    def compute_advantages(self, rewards, values, next_values, dones):
        advantages = rewards + self.gamma * next_values * (1.0 - dones) - values
        advantages = (advantages - np.mean(advantages)) / (np.std(advantages) + 1e-8)
        return advantages

    def compute_actor_loss(self, states, actions, advantages):
        probs = self.actor(states)
        action_dist = tfp.distributions.Categorical(probs=probs, dtype=tf.float32)
        log_probs = action_dist.log_prob(actions)
        actor_loss = -tf.reduce_mean(log_probs * advantages)

        # entropy regularization
        entropy = action_dist.entropy() 
        actor_loss -= self.entropy_beta * tf.reduce_mean(entropy)
        return actor_loss

        
    def train(self, states, actions, rewards, next_states, dones):
        states = np.array(states, dtype=np.float32)
        actions = np.array(actions, dtype=np.int32)
        rewards = np.array(rewards, dtype=np.float32)
        next_states = np.array(next_states, dtype=np.float32)
        dones = np.array(dones, dtype=np.float32)

        with tf.GradientTape() as tape1, tf.GradientTape() as tape2:
            values = self.critic(states)
            next_values = self.critic(next_states)
            advantages = self.compute_advantages(rewards, values, next_values, dones)
            actor_loss = self.compute_actor_loss(states, actions,advantages)
            critic_loss = tf.reduce_mean(tf.square(advantages))

        # compute gradients 
        actor_grads = tape1.gradient(actor_loss, self.actor.model.trainable_variables)
        critic_grads = tape2.gradient(critic_loss, self.critic.model.trainable_variables)

        # apply gradients to the models
        self.actor.optimizer.apply_gradients(zip(actor_grads, self.actor.model.trainable_variables))
        self.critic.optimizer.apply_gradients(zip(critic_grads, self.critic.model.trainable_variables))
        return actor_loss.numpy(), critic_loss.numpy()
    
    def compute_gradients(self, states, actions, rewards, next_states, dones):
        states = np.array(states, dtype=np.float32)
        actions = np.array(actions, dtype=np.int32)
        rewards = np.array(rewards, dtype=np.float32)
        next_states = np.array(next_states, dtype=np.float32)
        dones = np.array(dones, dtype=np.float32)

        with tf.GradientTape() as tape1, tf.GradientTape() as tape2:
            values = self.critic(states)
            next_values = self.critic(next_states)
            advantages = self.compute_advantages(rewards, values, next_values, dones)
            actor_loss = self.compute_actor_loss(states, actions, advantages)
            critic_loss = tf.reduce_mean(tf.square(advantages))

        # compute gradients 
        actor_grads = tape1.gradient(actor_loss, self.actor.model.trainable_variables)
        critic_grads = tape2.gradient(critic_loss, self.critic.model.trainable_variables)

        # clip gradients 
        actor_grads = [tf.clip_by_norm(grad, self.grad_clip_norm) \
                        if grad is not None else None for grad in actor_grads ]
        critic_grads = [tf.clip_by_norm(grad, self.grad_clip_norm) \
                        if grad is not None else None for grad in critic_grads]  

        # Check for NaN gradients
        actor_has_nan = any(tf.reduce_any(tf.math.is_nan(grad)) for grad in actor_grads if grad is not None)
        critic_has_nan = any(tf.reduce_any(tf.math.is_nan(grad)) for grad in critic_grads if grad is not None)


        if actor_has_nan or critic_has_nan:
            print(f"NaN gradients detected! "
                  f"Actor NaN: {actor_has_nan}, Critic NaN: {critic_has_nan}, "
                  f"Actor Loss: {actor_loss:.4f}, Critic Loss: {critic_loss:.4f}")
            # Skip sending gradients to avoid corrupting global network
            return None, None, None, None

        return actor_loss.numpy(), critic_loss.numpy(), actor_grads, critic_grads
    
    def apply_gradients(self, actor_grads, critic_grads):
        self.actor.optimizer.apply_gradients(zip(actor_grads, self.actor.model.trainable_variables))
        self.critic.optimizer.apply_gradients(zip(critic_grads, self.critic.model.trainable_variables))

    
    def save_weights(self, actor_wt_file='a2c_actor.weights.h5', 
                    critic_wt_file='a2c_critic.weights.h5'):
        self.actor.save_weights(actor_wt_file)
        self.critic.save_weights(critic_wt_file)
        
    def load_weights(self, actor_wt_file='a2c_actor.weights.h5', 
                     critic_wt_file='a2c_critic.weights.h5'):
        self.actor.load_weights(actor_wt_file)
        self.critic.load_weights(critic_wt_file)

    def get_weights(self):
        return self.actor.model.get_weights(), self.critic.model.get_weights()

    def set_weights(self, actor_weights, critic_weights):
        self.actor.model.set_weights(actor_weights)
        self.critic.model.set_weights(critic_weights)