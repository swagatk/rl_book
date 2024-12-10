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

    def __call__(self, states, target=False):
        if not target: # return probabilities
            pi = tf.squeeze(self.model(states))
        else:
            pi = tf.squeeze(self.target(states))
        return pi
    
    def train(self, state, action, advantage):
        with tf.GradientTape() as tape:
            actor_wts = self.model.trainable_variables
            pi = self.model(state)
            action_dist = tfp.distributions.Categorical(probs=pi)
            log_prob = action_dist.log_prob(action)
            actor_loss = -log_prob * advantage
            actor_grad = tape.gradient(actor_loss, actor_wts)
        self.optimizer.apply_gradients(zip(actor_grad, actor_wts))
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

    def __call__(self, states, target=False):
        if not target:
            value = tf.squeeze(self.model(states))
        else:
            value = tf.squeeze(self.target(states))
        return value

    def train(self, states, rewards, next_states, dones):
        with tf.GradientTape() as tape:
            critic_wts = self.model.trainable_variables
            values = self.model(states)
            target_values = self.model(next_states)
            y = rewards + self.gamma * (1 - dones) * target_values
            critic_loss = tf.math.reduce_mean(tf.square(y - values)) # TD error
            critic_grads = tape.gradient(critic_loss, critic_wts)
        self.optimizer.apply_gradients(zip(critic_grads, critic_wts))
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
        action_dist = tfp.distributions.Categorical(probs=pi)
        action = action_dist.sample()
        return action.numpy()


    def train(self, state, action, reward, next_state, done):
        state = tf.expand_dims(tf.convert_to_tensor(state, dtype=tf.float32), axis=0)
        action= tf.convert_to_tensor(action, dtype=tf.float32)
        reward = tf.convert_to_tensor(reward, dtype=tf.float32)
        next_state = tf.expand_dims(tf.convert_to_tensor(next_state, dtype=tf.float32), axis=0)
        done = tf.convert_to_tensor(done, dtype=tf.float32)
        value = self.critic(state)
        next_value = self.critic(next_state)
        td_target = reward + self.gamma * next_value * (1 - done)
        advantage = td_target - value
        a_loss = self.actor.train(state, action, advantage)
        c_loss = self.critic.train(state, reward, next_state, done)
        return a_loss, c_loss
        

    def save_weights(self, actor_wt_file='actor.weights.h5', 
                    critic_wt_file='critic.weights.h5',
                    buffer_file=None):
        self.actor.save_weights(actor_wt_file)
        self.critic.save_weights(critic_wt_file)
        if buffer_file is not None:
            self.buffer.save(buffer_file)
        
    def load_weights(self, actor_wt_file='actor.weights.h5', 
                     critic_wt_file='critic.weights.h5',
                    buffer_file=None):
        self.actor.load_weights(actor_wt_file)
        self.critic.load_weights(critic_wt_file)
        if buffer_file is not None:
            self.buffer.load(buffer_file)