'''
A2C implementation with a single worker.
- In this case, discounted returns is used as the target value for computing TD error. 
- the advance is estimated using the TD error.
- It has same performance as the other a2c implementation that computes advantages 
    using rewards, values, next_values, and dones.
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
    

    def compute_actor_loss(self, states, actions, td_error):
        probs = self.model(states)
        probability = []
        log_probability = []
        for pb, a in zip(probs, actions):
            dist = tfp.distributions.Categorical(probs=pb, dtype=tf.float32)
            log_prob = dist.log_prob(a)
            prob = dist.prob(a)
            probability.append(prob)
            log_probability.append(log_prob)

        td_error = td_error.numpy()
        p_loss = []
        e_loss = []
        for pb, t, lpb in zip(probability, td_error, log_probability):
            t = tf.constant(t, dtype=tf.float32)
            policy_loss = tf.math.multiply(lpb, t)
            entropy_loss = tf.math.negative(tf.math.multiply(pb, lpb))
            p_loss.append(policy_loss)
            e_loss.append(entropy_loss)

        p_loss = tf.stack(p_loss)
        e_loss = tf.stack(e_loss)
        p_loss = tf.reduce_mean(p_loss)
        e_loss = tf.reduce_mean(e_loss)

        actor_loss = -p_loss - 0.0001 * e_loss
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
                 lr_a=1e-4, lr_c=1e-4, gamma=0.99,
                 a_model=None, c_model=None):
        self.lr_a = lr_a
        self.lr_c = lr_c
        
        self.gamma = gamma
        self.action_size = action_size
        self.obs_shape = obs_shape
        self.name = 'A2C'

        # create actor and critic networks
        self.actor = Actor(obs_shape, action_size, lr=self.lr_a,
                           model=a_model)
        self.critic = Critic(obs_shape, lr=self.lr_c,
                             gamma=self.gamma, model=c_model)

    def policy(self, state):
        state = tf.expand_dims(tf.convert_to_tensor(state, dtype=tf.float32), axis=0)
        pi = self.actor(state)
        action_probs = tfp.distributions.Categorical(probs=pi)
        action = action_probs.sample()
        return action.numpy()
    
        
    def compute_discounted_rewards(self, states, actions, rewards):
        discounted_rewards = []
        sum_rewards = 0
        for r in reversed(rewards):
            sum_rewards = r + self.gamma * sum_rewards
            discounted_rewards.insert(0, sum_rewards)
        discounted_rewards = np.array(discounted_rewards)
        states = np.array(states, dtype=np.float32)
        actions = np.array(actions, dtype=np.int32)
        return states, actions, discounted_rewards

    def train(self, states, actions, rewards):
        states, actions, discnt_rewards = self.compute_discounted_rewards(states, actions, rewards)
        discnt_rewards = tf.convert_to_tensor(discnt_rewards, dtype=tf.float32)

        with tf.GradientTape() as tape1, tf.GradientTape() as tape2:
            values = self.critic(states)
            td_error = tf.math.subtract(discnt_rewards, values)
            actor_loss = self.actor.compute_actor_loss(states, actions, td_error)
            #critic_loss = 0.5 * tf.keras.losses.MeanSquaredError()(discnt_rewards, values) 
            critic_loss = tf.reduce_mean(tf.square(td_error))

        actor_grads = tape1.gradient(actor_loss, self.actor.model.trainable_variables)
        critic_grads = tape2.gradient(critic_loss, self.critic.model.trainable_variables)
        self.actor.optimizer.apply_gradients(zip(actor_grads, self.actor.model.trainable_variables))
        self.critic.optimizer.apply_gradients(zip(critic_grads, self.critic.model.trainable_variables))
        
        return actor_loss.numpy(), critic_loss.numpy()

    
    def save_weights(self, actor_wt_file='a2c_actor.weights.h5', 
                    critic_wt_file='a2c_critic.weights.h5'):
        self.actor.save_weights(actor_wt_file)
        self.critic.save_weights(critic_wt_file)
        
    def load_weights(self, actor_wt_file='a2c_actor.weights.h5', 
                     critic_wt_file='a2c_critic.weights.h5'):
        self.actor.load_weights(actor_wt_file)
        self.critic.load_weights(critic_wt_file)