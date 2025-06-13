import random
import numpy as np
import tensorflow as tf
import keras
from keras.optimizers import Adam
from buffer import ReplayBuffer, STBuffer
import os

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
        self.name = 'DQN'
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
        self.name = 'DQN_PER'

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
# DQN for Atari Environments  
##################

import imageio
from PIL import Image
import PIL.ImageDraw as ImageDraw
import matplotlib.pyplot as plt 
import cv2 
import sys 

def _label_with_episode_number(frame, episode_num, step_num):
    im = Image.fromarray(frame)
    drawer = ImageDraw.Draw(im)
    if np.mean(im) < 128: # for dark image
        text_color = (255, 255, 255)
    else:
        text_color = (0, 0, 0)
    drawer.text((im.size[0]/20, im.size[1]/18),
                f'Episode: {episode_num+1}, Steps: {step_num+1}',
                fill=text_color)
    return im

class DQNAtariAgent(DQNPERAgent):
    def __init__(self, obs_shape: tuple, n_actions: int,
                        buffer_size=2000, batch_size=24,
                        ddqn_flag=True, model=None, per_flag=True):
        self.per_flag = per_flag
        
        if self.per_flag:
            super().__init__(obs_shape, n_actions, buffer_size,
                        batch_size, ddqn_flag, model)
        else:
            DQNAgent.__init__(self, obs_shape, n_actions, buffer_size, 
                              batch_size, ddqn_flag, model)
        
    def experience_replay(self):
        if self.per_flag:
            super().experience_replay()
        else:
            DQNAgent.experience_replay(self)

    def preprocess(self, observation, x_crop=(1, 172), y_crop=None):
        assert len(self.obs_shape) == 3, "Observation must have 3 dimension (H, W, C)"
        output_shape = self.obs_shape[:-1] # all but last (H, W)
        # crop image
        if x_crop is not None and y_crop is not None:
            xlow, xhigh = x_crop
            ylow, yhigh = y_crop
            observation = observation[xlow:xhigh, ylow:yhigh]
        elif x_crop is not None and y_crop is None:
            xlow, xhigh = x_crop
            observation = observation[xlow:xhigh, :]
        elif x_crop is None and y_crop is not None:
                ylow, yhigh = y_crop
                observation = observation[:, ylow:yhigh]
        else:
            observation = observation
            
        # resize image
        observation = cv2.resize(observation, output_shape)
        
        # normalize image
        observation = observation / 255.  # normalize between 0 & 1
        return observation
            
    def train(self, env, max_episodes=300, 
          train_freq=1, copy_freq=1, filename=None, wtfile_prefix=None):
    
        if filename is not None:
            file = open(filename, 'w')
            
        if wtfile_prefix is not None:
            wt_filename = wtfile_prefix + '_best_model.weights.h5'
        else:
            wt_filename = 'best_model.weights.h5'

        tau = 0.01 if copy_freq < 10 else 1.0

        best_score, global_step_cnt = 0, 0
        scores, avg_scores, avg100_scores = [], [], []
        global_step_cnt = 0
        for e in range(max_episodes):
            state = env.reset() # with framestack wrapper
            #state = env.reset()[0] # without framestack wrapper
            state = self.preprocess(state)
            state = np.expand_dims(state, axis=0)
            done = False
            ep_reward = 0
            while not done:
                global_step_cnt += 1
                # take action
                action = self.get_action(state)
                # collect reward
                next_state, reward, done, _ = env.step(action) # with framestack wrapper
                #next_state, reward, done, _, _ = env.step(action) # without framestack wrapper
                next_state = self.preprocess(next_state) # (H, W, C)
                next_state = np.expand_dims(next_state, axis=0) # (B, H, W, C)
                # store experiences in eplay buffer
                self.store_experience(state, action, reward, next_state, done)
                state = next_state
                ep_reward += reward
                # train
                if global_step_cnt % train_freq == 0:
                    self.experience_replay()

                # update target model
                if global_step_cnt % copy_freq == 0:
                    self.update_target_model(tau=tau)
                # end of while-loop
            if ep_reward > best_score:
                self.save_model(wt_filename)
                best_score = ep_reward
            scores.append(ep_reward)
            avg_scores.append(np.mean(scores))
            avg100_scores.append(np.mean(scores[-100:]))
            if filename is not None:
                file.write(f'{e}\t{ep_reward}\t{np.mean(scores)}\t{np.mean(scores[-100:])}\n')
                file.flush()
                os.fsync(file.fileno())
            print(f'\re:{e}, ep_reward: {ep_reward}, avg_ep_reward: {np.mean(scores):.2f}', end="")
            sys.stdout.flush()
        # end of for loop
        print('\nEnd of training')
        if filename is not None:
            file.close()
            
    def validate(self, env, num_episodes=10, wt_file=None, gif_file=None):
        #ipdb.set_trace()
        if wt_file is not None:
            self.load_model(wt_file)
        frames, scores, steps = [], [], []
        for i in range(num_episodes):
            #sys.stdout.flush()
            state = env.reset()
            state = self.preprocess(state) # (H, W, C)
            state = np.expand_dims(state, axis=0) # (B, H, W, C)
            step = 0
            ep_reward = 0
            while True:
                step += 1
                if gif_file is not None and env.render_mode == 'rgb_array':
                    frame = env.render() # with framestack wrapper
                    #frame = env.render()[0] # without framestack wrapper
                    frames.append(_label_with_episode_number(frame, i, step))
                action = self.get_action(state, epsilon=0) # exploit
                next_state, reward, done, _ = env.step(action) # while using framestack wrapper
                #next_state, reward, done, _, _ = env.step(action) # not using framestack wrapper
                next_state = self.preprocess(next_state)
                next_state = np.expand_dims(next_state, axis=0)
                state = next_state
                ep_reward += reward
                if done:
                    scores.append(ep_reward)
                    if gif_file is not None and env.render_mode == 'rgb_array':
                        frame = env.render()
                        frames.append(_label_with_episode_number(frame, i, step))
                    break
            # while-loop ends here
            scores.append(ep_reward)
            steps.append(step)
            print(f'\repisode: {i}, reward: {ep_reward:.2f}, steps: {step}')
        # for-loop ends here
        if gif_file is not None:
            imageio.mimwrite(os.path.join('./', gif_file), frames, duration=1000/60)
        print('\nAverage episodic score: ', np.mean(scores))
        print('\nAverage episodic steps: ', np.mean(steps))