import random
import numpy as np
import tensorflow as tf
import keras
from keras.optimizers import Adam
from buffer import ReplayBuffer, STBuffer

class DQNAgent:
    def __init__(self, obs_shape: tuple, n_actions: int,
               buffer_size=2000, batch_size=24,
                 ddqn_flag=True, model=None):

        self.obs_shape = obs_shape  # shape: tuple
        self.action_size = n_actions  # number of discrete action state (int)
        self.ddqn = ddqn_flag      # choose between DQN & DDQN
        # hyper parameters for DQN
        self.gamma =  0.9         # discount factor
        self.epsilon = 1.0        # explore rate
        self.epsilon_decay = 0.99
        self.epsilon_min = 0.01
        self.batch_size = batch_size
        self.buffer_size = buffer_size  # replay buffer size
        self.train_start = 1000     # minimum buffer size to start training
        self.learning_rate = 0.001
        # create a replay buffer to store experiences
        self.memory = ReplayBuffer(self.buffer_size)
        # create main model and target model
        if model is None:
            self.model = self._build_model()
            if self.ddqn:
                self.target_model = self._build_model()
        else:
            self.model = model
            if self.ddqn:
                self.target_model = tf.keras.models.clone_model(model)
        self.model.summary()
        # initialize target model
        if self.ddqn:
            self.target_model.set_weights(self.model.get_weights())
        
    def _build_model(self):
        # approximate Q-function with a Neural Network
        model = keras.Sequential([
        keras.layers.Dense(24, input_shape=self.obs_shape, activation='relu',
                kernel_initializer='he_uniform'),
        keras.layers.Dense(24, activation='relu',
                       kernel_initializer='he_uniform'),
        keras.layers.Dense(self.action_size, activation='linear',
                kernel_initializer='he_uniform')
        ])
        model.summary()
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
        return model

    def update_target_model(self, tau=0.01):
        if self.ddqn: # applicable only for double dqn
            model_weights = self.model.get_weights()
            target_weights = self.target_model.get_weights()
            # Ensure shapes match
            if len(model_weights) != len(target_weights):
                raise ValueError('Model and Target should have same number of items')

            # update weights layer-by-layer using Polyak Averaging
            new_weights = []
            for w, w_dash in zip(model_weights, target_weights):
                new_w = tau * w + (1 - tau) * w_dash
                new_weights.append(new_w)
            self.target_model.set_weights(new_weights)

    def get_action(self, state, epsilon=None):
        # get action using epsilon-greedy policy
        if epsilon is None:
            epsilon = self.epsilon
        if np.random.rand() <= epsilon: # explore
            return random.randrange(self.action_size)
        else:
            q_value = self.model.predict(state, verbose=0)  # exploit
            return np.argmax(q_value[0])

    def store_experience(self, state, action, reward, next_state, done):
        # save sample <s, a, r, s'>. into replay memory
        self.memory.add((state, action, reward, next_state, done)) 

    def get_target_q_value(self, next_states): # batch input
        q_values_ns = self.model.predict(next_states, verbose=0) # Q(s',a')
        if self.ddqn: # DDQN algo
            # main model is used for action selection
            max_actions = np.argmax(q_values_ns, axis=1)
            # target model is used for action evaluation
            target_q_values_ns = self.target_model.predict(next_states, verbose=0)
            max_q_values = target_q_values_ns[range(len(target_q_values_ns)), max_actions]
        else: # DQN
            max_q_values = np.amax(q_values_ns, axis=1)
        return max_q_values

    def experience_replay(self):
        if len(self.memory) < self.train_start:
            return
        # sample experiences from replay buffer
        batch_size = min(self.batch_size, len(self.memory))
        mini_batch = self.memory.sample(self.batch_size)

        states = np.zeros((self.batch_size, *self.obs_shape))
        next_states = np.zeros((self.batch_size, *self.obs_shape))
        actions = np.zeros((self.batch_size, 1))
        rewards = np.zeros((self.batch_size, 1))
        dones = np.zeros((self.batch_size, 1))

        for i in range(len(mini_batch)):
            states[i] = mini_batch[i][0]
            actions[i] = mini_batch[i][1]
            rewards[i] = mini_batch[i][2]
            next_states[i] = mini_batch[i][3]
            dones[i]  = mini_batch[i][4]

        q_values_cs = self.model.predict(states, verbose=0)
        max_q_values_ns = self.get_target_q_value(next_states)
        
        for i in range(len(q_values_cs)):
            action = actions[i].astype(int)[0]
            done = dones[i].astype(bool)[0]
            reward = rewards[i][0] 
            if done:
                q_values_cs[i][action] = reward
            else:
                q_values_cs[i][action] = reward + self.gamma * max_q_values_ns[i]
                
        # train the Q network
        self.model.fit(np.array(states),
                       np.array(q_values_cs),
                       batch_size = batch_size,
                       epochs = 1,
                       verbose = 0)

        # decay epsilon over time
        self.update_epsilon()

    def update_epsilon(self):
        # decay epsilon durin training 
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def save_model(self, filename: str):
        if filename.lower().endswith(".weights.h5"):
            self.model.save_weights(filename)
        else:
            raise ValueError("filename must have '.weights.h5' as extension")
        
    def load_model(self, filename: str):
        if filename.lower().endswith(".weights.h5"):
            self.model.load_weights(filename)
        else:
            raise ValueError("filename must have '.weights.h5' as extension")
    
    def get_epsilon(self):
        return self.epsilon 

# end of class
#############

class DQNPERAgent(DQNAgent):
    '''
    DQN with Priority Experience Replay
    '''
    def __init__(self, obs_shape: tuple, n_actions: int,
                        buffer_size=2000, batch_size=24,
                        ddqn_flag=True, model=None):
        super().__init__(obs_shape, n_actions, buffer_size,
                        batch_size, ddqn_flag, model)

        # uses a sumtree Buffer
        self.memory = STBuffer(capacity=buffer_size)

    def experience_replay(self):
        if len(self.memory) < self.train_start:
            return
        batch_size = min(self.batch_size, len(self.memory))
        tree_idx, mini_batch = self.memory.sample(self.batch_size)

        states = np.zeros((self.batch_size, *self.obs_shape))
        next_states = np.zeros((self.batch_size, *self.obs_shape))
        actions = np.zeros((self.batch_size, 1))
        rewards = np.zeros((self.batch_size, 1))
        dones = np.zeros((self.batch_size, 1))

        for i in range(len(mini_batch)):
            states[i] = mini_batch[i][0]
            actions[i] = mini_batch[i][1]
            rewards[i] = mini_batch[i][2]
            next_states[i] = mini_batch[i][3]
            dones[i]  = mini_batch[i][4]

        q_values_cs = self.model.predict(states, verbose=0)
        q_values_cs_old = np.array(q_values_cs).copy() # deep copy
        max_q_values_ns = self.get_target_q_value(next_states)


        for i in range(len(q_values_cs)):
            action = actions[i].astype(int)[0] # check
            done = dones[i].astype(bool)[0] # check
            reward = rewards[i][0] # check
            if done:
                q_values_cs[i][action] = reward
            else:
                q_values_cs[i][action] = reward + self.gamma * max_q_values_ns[i]

        # update experience priorities
        indices = np.arange(self.batch_size, dtype=np.int32)
        actions = actions[:,0].astype(int)
        absolute_errors = np.abs(q_values_cs_old[indices, actions] - \
                                q_values_cs[indices, actions])
        # update sample priorities
        self.memory.batch_update(tree_idx, absolute_errors)

        # train the Q network
        self.model.fit(np.array(states),
                    np.array(q_values_cs),
                    batch_size = batch_size,
                    epochs = 1,
                    verbose = 0)

        # decay epsilon over time
        self.update_epsilon()

###################
#   