'''
Actor-Critic agent for discrete action space 

'''
import tensorflow as tf
import numpy as np
import tensorflow_probability as tfp
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
    
    def compute_actor_loss(self, states, actions, td_error):
        # computes actor loss for a batch of states, actions, and advantages
        pi = self.model(states)
        action_dist = tfp.distributions.Categorical(probs=pi, dtype=tf.float32)
        log_prob = action_dist.log_prob(actions)
        actor_loss = -log_prob * td_error
        return tf.math.reduce_mean(actor_loss)
    

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


#################

class ACAgent():
    def __init__(self, obs_shape, action_size, 
                lr_a=1e-4, lr_c=1e-4, gamma=0.99,
                a_model=None, c_model=None):
        self.obs_shape = obs_shape
        self.action_size = action_size
        self.gamma = gamma
        self.lr_a = lr_a
        self.lr_c = lr_c
        self.name = 'actor-critic'

        # actor model
        self.actor = Actor(self.obs_shape, self.action_size,
                          lr=self.lr_a, model=a_model)
        # critic model
        self.critic = Critic(self.obs_shape, lr=self.lr_c, 
                             gamma=self.gamma, model=c_model)

    def policy(self, state):
        state = tf.expand_dims(tf.convert_to_tensor(state, dtype=tf.float32), axis=0)
        pi = self.actor(state) # action probabilities
        action_dist = tfp.distributions.Categorical(probs=pi, dtype=tf.float32)
        action = action_dist.sample()
        return int(action.numpy())


    def train(self, state, action, reward, next_state, done):
        state = tf.expand_dims(tf.convert_to_tensor(state, dtype=tf.float32), axis=0)
        action= tf.convert_to_tensor(action, dtype=tf.float32)
        reward = tf.convert_to_tensor(reward, dtype=tf.float32)
        next_state = tf.expand_dims(tf.convert_to_tensor(next_state, dtype=tf.float32), axis=0)
        done = tf.convert_to_tensor(done, dtype=tf.float32)
        with tf.GradientTape() as tape1, tf.GradientTape() as tape2:
            value = self.critic(state)
            next_value = self.critic(next_state)
            td_target = reward + self.gamma * next_value * (1 - int(done))
            td_error = td_target - value
            a_loss = self.actor.compute_actor_loss(state, action, td_error)
            c_loss = tf.math.reduce_mean(tf.square(td_target - value))
        actor_grads = tape1.gradient(a_loss, self.actor.model.trainable_variables)
        critic_grads = tape2.gradient(c_loss, self.critic.model.trainable_variables)
        self.actor.optimizer.apply_gradients(zip(actor_grads, self.actor.model.trainable_variables))
        self.critic.optimizer.apply_gradients(zip(critic_grads, self.critic.model.trainable_variables))
        return a_loss, c_loss
        

    def save_weights(self, actor_wt_file='ac_actor.weights.h5', 
                    critic_wt_file='ac_critic.weights.h5'):
        self.actor.save_weights(actor_wt_file)
        self.critic.save_weights(critic_wt_file)
        
    def load_weights(self, actor_wt_file='ac_actor.weights.h5', 
                     critic_wt_file='ac_critic.weights.h5'):
        self.actor.load_weights(actor_wt_file)
        self.critic.load_weights(critic_wt_file)