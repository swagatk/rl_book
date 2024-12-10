import tensorflow as tf
import numpy as np
import tensorflow_probability as tfp

######################
## Actor network
class PPOActor():
    def __init__(self, obs_shape, action_size,
                 learning_rate=0.0003, 
                 action_upper_bound=1.0,
                 epsilon=0.2, lmbda=0.5, kl_target=0.1, 
                 beta=0.1, entropy_coeff=0.2,
                 critic_loss_coeff=0.1,
                 grad_clip=10.0,
                 method='clip', 
                model=None):
        self.obs_shape = obs_shape
        self.action_size = action_size
        self.action_ub = action_upper_bound
        self.lr = learning_rate
        self.epsilon = epsilon # clip on ratio
        self.lam = lmbda # required for penalty method
        self.beta = beta # kl penalty coefficient
        self.entropy_coeff = entropy_coeff
        self.c_loss_coeff = critic_loss_coeff
        self.method = method # choose between 'clip' and 'penalty'
        self.kl_target = kl_target
        self.kl_value = 0 # to store most recent kld value
        self.grad_clip = grad_clip # applying gradient clipping
        
        # create actor model
        if model is None:
            self.model = self._build_model()
        else:
            self.model = tf.keras.models.clone_model(model)
        self.optimizer = tf.keras.optimizers.Adam(self.lr)
        
        # additional parameters
        logstd = tf.Variable(np.zeros(shape=(self.action_size, )), dtype=np.float32)
        self.model.logstd = logstd
        self.model.trainable_variables.append(logstd)
        
    def _build_model(self):
        last_init = tf.random_uniform_initializer(minval=-0.01, maxval=0.01)
        state_input = tf.keras.layers.Input(shape=self.obs_shape)
        x = tf.keras.layers.Dense(128, activation='relu')(state_input)
        x = tf.keras.layers.Dense(64, activation='relu')(x)
        x = tf.keras.layers.Dense(64, activation='relu')(x)
        net_out = tf.keras.layers.Dense(self.action_size, activation='tanh',
                                       kernel_initializer=last_init)(x)
        net_out = net_out * self.action_ub
        model = tf.keras.models.Model(state_input, net_out, name='actor')
        model.summary()
        return model
    
    def __call__(self, state):
        # input is a tensor
        mean = tf.squeeze(self.model(state))
        std = tf.squeeze(tf.exp(self.model.logstd))
        return mean, std # return tensor
    
    def train(self, state_batch, action_batch, advantages, old_pi, c_loss):
        #ipdb.set_trace()
        with tf.GradientTape() as tape:
            mean = tf.squeeze(self.model(state_batch))
            std = tf.squeeze(tf.exp(self.model.logstd))
            pi = tfp.distributions.Normal(mean, std)
            ratio = tf.exp(pi.log_prob(tf.squeeze(action_batch)) -
                          old_pi.log_prob(tf.squeeze(action_batch))) # r = pi/pi_old
            if ratio.ndim > advantages.ndim:
                ratio = tf.reduce_mean(ratio, axis=-1) # convert into 1-D to match with advantage shape
            surr_obj = ratio * advantages # surrogate objective function
            # current kl divergence (kld) value
            kld = tfp.distributions.kl_divergence(old_pi, pi)
            if kld.ndim > advantages.ndim:
                kld = tf.reduce_mean(kld, axis=-1)
            self.kl_value = tf.reduce_mean(kld)
            entropy = tf.reduce_mean(pi.entropy()) # entropy
            if self.method == 'penalty':
                actor_loss = -(tf.reduce_mean(surr_obj - self.beta * kld)) # maximize
            elif self.method == 'clip':
                l_clip = tf.reduce_mean(
                        tf.minimum(surr_obj, tf.clip_by_value(ratio,
                        1.-self.epsilon, 1.+self.epsilon) * advantages))
                actor_loss = -(l_clip - self.c_loss_coeff * c_loss  + \
                              self.entropy_coeff * entropy)
            else:
                raise ValueError('invalid option for PPO method')
            actor_weights = self.model.trainable_variables
            actor_grad = tape.gradient(actor_loss, actor_weights)
            if self.grad_clip is not None:
                actor_grad = [tf.clip_by_value(grad, -1 * self.grad_clip, 
                                                self.grad_clip) for grad in actor_grad]
        #outside gradient tape
        self.optimizer.apply_gradients(zip(actor_grad, actor_weights))
        return actor_loss.numpy()
                
    def update_beta(self):
        # update beta after each epoch
        if self.kl_value < self.kl_target / 1.5:
            self.beta /= 2.
        elif self.kl_value > self.kl_target * 1.5:
            self.beta *= 2.
            
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
        
####################
## PPO Critic Network

class PPOCritic():
    def __init__(self, obs_shape, action_size,
                 learning_rate=0.0003,
                 gamma=0.99,
                 grad_clip = None,
                 model=None):
        self.lr = learning_rate
        self.obs_shape = obs_shape
        self.action_size = action_size
        self.gamma = gamma
        self.grad_clip = grad_clip
        
        if model is None:
            self.model = self._build_model()
        else:
            self.model = tf.keras.models.clone_model(model)
        self.optimizer = tf.keras.optimizers.Adam(self.lr)
        
    def __call__(self, state):
        # input is a tensor
        value = tf.squeeze(self.model(state))
        return value
    
    def _build_model(self):
        state_input = tf.keras.layers.Input(shape=self.obs_shape)
        out = tf.keras.layers.Dense(64, activation="relu")(state_input)
        out = tf.keras.layers.Dense(64, activation="relu")(out)
        out = tf.keras.layers.Dense(64, activation="relu")(out)
        net_out = tf.keras.layers.Dense(1)(out)
        # Outputs single value for give state-action
        model = tf.keras.models.Model(inputs=state_input, outputs=net_out)
        model.summary()
        return model
        
        
    def train(self, state_batch, disc_rewards):
        with tf.GradientTape() as tape:
            critic_weights = self.model.trainable_variables
            critic_value = tf.squeeze(self.model(state_batch))
            critic_loss = tf.math.reduce_mean(tf.square(disc_rewards - critic_value))
            critic_grad = tape.gradient(critic_loss, critic_weights)
            if self.grad_clip is not None:
                critic_grad = [tf.clip_by_value(grad, -1.0 * self.grad_clip,
                                          self.grad_clip) for grad in critic_grad]
        # outside the gradient tape
        self.optimizer.apply_gradients(zip(critic_grad, critic_weights))
        return critic_loss.numpy()
    
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
        

#####################
## PPO Agent

class PPOAgent:
    def __init__(self, obs_shape, action_size, batch_size,
                 action_upper_bound=1.0,
                 lr_a=1e-3, lr_c=1e-3,
                 gamma=0.99,            # discount factor
                 lmbda=0.5,            # required for GAE
                 beta=0.01,              # KL penalty coefficient
                 epsilon=0.2,           # action clip boundary
                 kl_target=0.01,        # required for KL penalty method
                 entropy_coeff=0.01,     # entropy coefficient
                 c_loss_coeff=0.01,      # critic loss coefficient
                 grad_clip=None,
                 method='clip',         # choose between 'clip' & 'penalty'
                 actor_model=None,
                 critic_model=None):
        self.name='ppo'
        self.obs_shape = obs_shape
        self.action_size = action_size
        self.actor_lr = lr_a
        self.critic_lr = lr_c
        self.batch_size = batch_size
        self.gamma = gamma # discount factor
        self.action_upper_bound = action_upper_bound
        self.epsilon = epsilon # clip boundary for prob ratio
        self.lmbda = lmbda # required for GAE
        self.initial_beta = beta # required for penalty method
        self.kl_target = kl_target # required for updating beta
        self.method = method # choose between 'clip' & 'penalty'
        self.c_loss_coeff = c_loss_coeff
        self.entropy_coeff = entropy_coeff
        self.grad_clip = grad_clip # apply gradient clipping
        

        # Actor Model
        self.actor = PPOActor(self.obs_shape, self.action_size,
                              learning_rate=self.actor_lr, 
                              action_upper_bound=self.action_upper_bound,
                              epsilon=self.epsilon, 
                              lmbda=self.lmbda,
                              kl_target=self.kl_target, 
                              beta=self.initial_beta, 
                              entropy_coeff=self.entropy_coeff,
                              critic_loss_coeff=self.c_loss_coeff,
                              method=self.method, 
                              grad_clip=self.grad_clip,
                              model=actor_model)
        # Critic Model
        self.critic = PPOCritic(self.obs_shape, self.action_size,
                               learning_rate=self.critic_lr,
                                gamma=self.gamma,
                                grad_clip=self.grad_clip,
                                model=critic_model)

    def policy(self, state, greedy=False):
        tf_state = tf.expand_dims(tf.convert_to_tensor(state), axis=0)
        mean, std = self.actor(tf_state)
        if greedy:
            action = mean
        else:
            pi = tfp.distributions.Normal(mean, std)
            action = pi.sample()
            action = tf.reshape(action, shape=(self.action_size, ))
        valid_action = tf.clip_by_value(action, -self.action_upper_bound, 
                                        self.action_upper_bound)
        return valid_action.numpy()
    
    def train(self, states, actions, rewards, next_states, dones, epochs=20):
        states = tf.convert_to_tensor(states, dtype=tf.float32)
        next_states = tf.convert_to_tensor(next_states, dtype=tf.float32)
        actions = tf.convert_to_tensor(actions, dtype=tf.float32)
        rewards = tf.convert_to_tensor(rewards, dtype=tf.float32)
        dones = tf.convert_to_tensor(dones, dtype=tf.float32)
        
        # compute advantage & discounted returns
        target_values, advantages = self.compute_advantages(states, rewards, next_states, dones)
        target_values = tf.convert_to_tensor(target_values, dtype=tf.float32)
        advantages = tf.convert_to_tensor(advantages, dtype=tf.float32)
        

        # current action probability distribution
        mean, std = self.actor(states)
        pi = tfp.distributions.Normal(mean, std)
        n_split = len(rewards) // self.batch_size
        assert n_split > 0, 'buffer length must be greater than batch_size'
        indexes = np.arange(n_split, dtype=int)
        
        
        # training
        a_loss_list, c_loss_list, kl_list = [], [], []
        for _ in range(epochs):
            np.random.shuffle(indexes) # this is the change
            for i in indexes:
                old_pi = pi[i * self.batch_size: (i+1) * self.batch_size]
                s_split = tf.gather(states, indices=np.arange(i * self.batch_size, (i+1) * self.batch_size), axis=0)
                a_split = tf.gather(actions, indices=np.arange(i * self.batch_size, (i+1) * self.batch_size), axis=0)
                tv_split = tf.gather(target_values, indices=np.arange(i * self.batch_size, (i+1) * self.batch_size), axis=0)
                adv_split = tf.gather(advantages, indices=np.arange(i * self.batch_size, (i+1) * self.batch_size), axis=0)
                
                # update critic
                cl = self.critic.train(s_split, tv_split)
                c_loss_list.append(cl)
                
                # update actor
                al = self.actor.train(s_split, a_split, adv_split, old_pi, cl)
                a_loss_list.append(al)
                kl_list.append(self.actor.kl_value)
                
            # update lambda once in each epoch
            if self.method == 'penalty':
                self.actor.update_beta()
                
        # end of epoch loop
        actor_loss = np.mean(a_loss_list)
        critic_loss = np.mean(c_loss_list)
        kld_mean = np.mean(kl_list)
        return actor_loss, critic_loss, kld_mean
    
    def compute_advantages(self, states, rewards, next_states, dones):
        # input/output are tensors
        s_values = self.critic(states)
        ns_values = self.critic(next_states)
        
        adv = np.zeros_like(s_values) # advantage should have same shape as the values
        returns = np.zeros_like(s_values)
        
        discount = self.gamma
        lmbda = self.lmbda
        returns_current = ns_values[-1] # last value
        g = 0 # GAE
        for i in reversed(range(len(rewards))):
            gamma = discount * (1. - dones[i])
            td_error = rewards[i] + gamma * ns_values[i] - s_values[i]
            g = td_error + gamma * lmbda * g
            returns_current = rewards[i] + gamma * returns_current
            adv[i] = g
            returns[i] = returns_current
        adv = (adv - np.mean(adv)) / (np.std(adv) + 1e-10)
        return returns, adv
    
    def save_weights(self, actor_wt_file='actor.weights.h5', 
                    critic_wt_file='critic.weights.h5'):
        self.actor.save_weights(actor_wt_file)
        self.critic.save_weights(critic_wt_file)

    def load_weights(self, actor_wt_file='actor.weights.h5', 
                     critic_wt_file='critic.weights.h5'):
        self.actor.load_weights(actor_wt_file)
        self.critic.load_weights(critic_wt_file)
        print('Model weights are loaded.')
        
    @property
    def penalty_coefficient(self):
        # returns penalty coefficienty
        return self.actor.beta

##########