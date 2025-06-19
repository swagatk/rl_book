'''
LunarLander-v2 with Actor-Critic
'''
import gymnasium as gym
import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
from actor_critic import ACAgent
import os
import wandb
import sys

sys.path.append("/Share/rl_book/chap06") 
from utils import validate

train = True
##########

# create actor & Critic models
def create_actor_model(obs_shape, n_actions):
    s_input = tf.keras.layers.Input(shape=obs_shape)
    x = tf.keras.layers.Dense(128, activation='relu')(s_input)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    a = tf.keras.layers.Dense(n_actions, activation='softmax')(x)
    model = tf.keras.models.Model(s_input, a, name='actor_network')
    model.summary()
    return model

def create_critic_model(obs_shape):
    s_input = tf.keras.layers.Input(shape=obs_shape)
    x = tf.keras.layers.Dense(128, activation='relu')(s_input)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    v = tf.keras.layers.Dense(1, activation=None)(x)
    model = tf.keras.models.Model(s_input, v, name='critic_network')
    model.summary()
    return model

###############
def ac_train(env, agent, max_episodes=10000, log_freq=50, 
            max_score=None, min_score=None,
            stop_score=499, 
            filename=None, wandb_log=False):
    
    print('Environment name: ', env.spec.name)
    print('RL Agent name:', agent.name)
    
    assert isinstance(env.action_space, gym.spaces.Discrete),\
                "AC Agent only for discrete action spaces"

    if filename is not None:
        file = open(filename, 'w')

    if wandb_log:
        run = wandb.init(entity='swagatk', project=env.spec.id, 
                         config={
                             'lr_a': agent.lr_a,
                             'lr_c': agent.lr_c,
                             'gamma': agent.gamma,
                             'agent': agent.name})

    ep_scores = []
    best_score = -np.inf 
    for e in range(max_episodes):
        done = False
        state = env.reset()[0]
        ep_score = 0
        a_losses, c_losses = [], []
        while not done:
            action = agent.policy(state)
            next_state, reward, done, _, _ = env.step(action)
            a_loss, c_loss = agent.train(state, action, reward, next_state, done)
            a_losses.append(a_loss)
            c_losses.append(c_loss)
            state = next_state
            ep_score += reward


            if max_score is not None and ep_score >= max_score:
                done = True
            if min_score is not None and ep_score <= min_score:
                done = True

        # while loop ends here
        ep_scores.append(ep_score)
        if ep_score > best_score:
            best_score = ep_score
            if best_score > stop_score:
                agent.save_weights()
                print('Best score: {:.2f} at episode {}. Model Saved!'.format(best_score, e))
        
        if filename is not None:
            file.write(f'{e}\t{ep_score}\t{np.mean(ep_scores)}\t{a_loss}\t{c_loss}\n')
            file.flush()
            os.fsync(file.fileno())
        if e % log_freq == 0:
            print(f'e:{e}, ep_score:{ep_score:.2f}, avg_ep_score:{np.mean(ep_scores):.2f},\
            avg100score:{np.mean(ep_scores[-100:]):.2f}, best_score:{best_score:.2f}')
        
        if wandb_log:
            wandb.log({
                'episode': e,
                'ep_score': ep_score, 
                'avg100score': np.mean(ep_scores[-100:]),
                'actor_loss': np.mean(a_losses),
                'critic_loss': np.mean(c_losses),
                'mean_score': np.mean(ep_scores),
                'best_score': best_score,
            })
        if e > 100 and np.mean(ep_scores[-100:]) > stop_score:
            print('The problem is solved in {} episodes'.format(e))
            break
    # for loop ends here
    print('Training completed.')
    if filename is not None:
        file.close()
    if wandb_log:
        run.finish()
    

##################

#########################
if __name__ == "__main__":

    env = gym.make('LunarLander-v3', render_mode='rgb_array')
    obs_shape = env.observation_space.shape
    action_size = env.action_space.n 

    print('Observation shape: ', obs_shape)
    print('Action Size: ', action_size)
    print('Max Episode steps: ', env.spec.max_episode_steps)

    actor = create_actor_model(obs_shape, action_size)
    critic = create_critic_model(obs_shape)

    # create an RL agent using above actor/critic models
    agent = ACAgent(obs_shape, action_size,
                    a_model=actor, c_model=critic)

    # train the RL agent on
    ac_train(env, agent, max_episodes=1500, min_score=-500, log_freq=100, max_score=500, stop_score=200, wandb_log=True)

    # validate the agent
    # agent.load_weights()  # load the best weights
    # validate(env, agent, num_episodes=10, gif_file='lunarlander_ac.gif',)
