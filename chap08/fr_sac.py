
"""
Solving the FetchReach-v3 environment using Soft Actor-Critic (SAC) algorithm.

In order to generate animation in validate() function, you need to install the following dependencies:
sudo apt update
sudo apt-get install libegl1
export MUJOCO_GL=egl
"""
import gymnasium as gym
import gymnasium_robotics
gym.register_envs(gymnasium_robotics)

import numpy as np
import wandb
import os
import sys
import tensorflow as tf
from PIL import Image, ImageDraw
import imageio

SAC2 = False
EXT_MODEL = False
EVAL = False
####################
if SAC2:
    from sac2 import SACAgent
else:
    from sac import SACAgent

# create actor & Critic models
def create_actor_model(obs_shape, n_actions):
    s_input = tf.keras.layers.Input(shape=obs_shape)
    x = tf.keras.layers.Dense(256, activation='relu')(s_input)
    x = tf.keras.layers.Dense(256, activation='relu')(s_input)
    x = tf.keras.layers.Dense(256, activation='relu')(x)
    mu = tf.keras.layers.Dense(n_actions, activation=None)(x)
    sigma = tf.keras.layers.Dense(n_actions, activation=None)(x)
    model = tf.keras.models.Model(s_input, [mu, sigma], name='actor_network')
    model.summary()
    return model

def create_critic_model(obs_shape, action_shape):
    s_input = tf.keras.layers.Input(shape=obs_shape)
    a_input = tf.keras.layers.Input(shape=action_shape)
    x = tf.keras.layers.Concatenate()([s_input, a_input])
    x = tf.keras.layers.Dense(256, activation='relu')(s_input)
    x = tf.keras.layers.Dense(256, activation='relu')(x)
    x = tf.keras.layers.Dense(256, activation='relu')(x)
    q = tf.keras.layers.Dense(1, activation=None)(x)
    model = tf.keras.models.Model([s_input, a_input], q, name='critic_network')
    model.summary()
    return model

def create_value_network(obs_shape):
    s_input = tf.keras.layers.Input(shape=obs_shape)
    x = tf.keras.layers.Dense(256, activation='relu')(s_input)
    x = tf.keras.layers.Dense(256, activation='relu')(x)
    x = tf.keras.layers.Dense(256, activation='relu')(x)
    v = tf.keras.layers.Dense(1, activation=None)(x)
    model = tf.keras.models.Model(s_input, v, name='value_network')
    model.summary()
    return model

#######################
def train_sac_agent(env, agent, num_episodes=1500, 
                    max_score=None, min_score=None, 
                    stop_score=-200, 
                    warmup_steps=1000,
                    ep_max_steps=None,
                    update_per_step=1,
                    log_freq=100,
                    train_freq=1,
                    filename=None, wandb_log=False):
    
    print('Environment name: ', env.spec.id)
    print('RL Agent name:', agent.name)
    
    if filename is not None:
        file = open(filename, 'w')

    if wandb_log:
        run = wandb.init(entity='swagatk', project=env.spec.id, 
            config={
                'lr_a': agent.actor_lr,
                'lr_c': agent.critic_lr,
                'lr_alpha': agent.alpha_lr,
                'buffer_size': agent.buffer_size,
                'batch_size': agent.batch_size,
                'gamma': agent.gamma,
                'agent': agent.name,
                'polyak': agent.polyak,
                })

    ep_scores = []
    best_score = -np.inf
    total_steps = 0
    a_loss, c_loss, alpha_loss = 0, 0, 0
    for e in range(num_episodes):
        done = False
        truncated = False
        obs = env.reset()[0]
        state = obs['observation']  # Extract observation from the info dict
        ep_score = 0
        ep_steps = 0
        c_losses, a_losses, alpha_losses = [], [], []
        if SAC2:
            v_losses = []
        while not done and not truncated:
            if total_steps < warmup_steps:
                action = env.action_space.sample()
            else:
                # Use the agent's policy to select an action
                action = agent.choose_action(state)

            
            if np.isnan(action).any() or np.isinf(action).any():
                raise ValueError("Bad action detected!", action)

            next_obs, reward, done, truncated, _ = env.step(action)
            next_state = next_obs['observation'] # Extract next observation from the info dict
            agent.store_transition(state, action, reward, next_state, done)

            state = next_state
            ep_score += reward
            ep_steps += 1
            total_steps += 1

            if max_score is not None and ep_score >= max_score:
                done = True
            if min_score is not None and ep_score <= min_score:
                done = True
            if ep_max_steps is not None and ep_steps >= ep_max_steps:
                done = True

            # train the agent
            if total_steps >= warmup_steps and total_steps % train_freq == 0:
                if SAC2:
                    vl, cl, al, alphal = agent.train()   
                    v_losses.append(vl)
                else:
                   cl, al, alphal = agent.train(update_per_step=update_per_step)   
                c_losses.append(cl)
                a_losses.append(al)
                alpha_losses.append(alphal)            


        # while loop ends here
        ep_scores.append(ep_score)
        c_loss = np.mean(c_losses) 
        a_loss = np.mean(a_losses) 
        alpha_loss = np.mean(alpha_losses)
        v_loss = np.mean(v_losses) if SAC2 else 0


        if filename is not None:
            file.write(f'{e}\t{ep_score}\t{np.mean(ep_scores)}\t{a_loss}\t{c_loss}\t{alpha_loss}\n')
            file.flush()
            os.fsync(file.fileno())

        if e % log_freq == 0:
            print(f'e:{e}, ep_score:{ep_score:.2f}, avg_ep_score:{np.mean(ep_scores):.2f},\
            avg100score:{np.mean(ep_scores[-100:]):.2f}, \
                best_score:{best_score:.2f}')
        
        if wandb_log:
            wandb.log({
                'episode': e,
                'ep_score': ep_score, 
                'avg100score': np.mean(ep_scores[-100:]),
                'actor_loss': a_loss, 
                'critic_loss': c_loss,
                'mean_score': np.mean(ep_scores),
                'best_score': best_score,
                'alpha_loss': alpha_loss,
                'v_loss': v_loss if SAC2 else 0
            })

        if ep_score > best_score:
            best_score = ep_score
            agent.save_weights(filename='fr_sac.weights.h5')
            print(f'Best Score: {ep_score:.2f}, episode: {e}. Model saved.')

        if np.mean(ep_scores[-100:]) >= stop_score:
            print('The problem is solved in {} episodes'.format(e))
            break
    # for loop ends here
    if filename is not None:
        file.close()
    if wandb_log:
        run.finish()

###############

def _label_with_episode_number(frame, episode_num, step_num):
    im = Image.fromarray(frame)
    drawer = ImageDraw.Draw(im)
    if np.mean(np.array(im)) < 128: # for dark image
        text_color = (255, 255, 255)
    else:
        text_color = (0, 0, 0)
    drawer.text((im.size[0]/20, im.size[1]/18),
                f'Episode: {episode_num+1}, Steps: {step_num+1}',
                fill=text_color)
    return im


def validate(env, agent, num_episodes=10, max_steps=None, gif_file=None):
    frames, scores, steps = [], [], []
    if gif_file:
        if env.render_mode != 'rgb_array':
            raise ValueError('To save a GIF, the environment must be in rgb_array render mode.')
        if not gif_file.endswith('.gif'):
            gif_file += '.gif'
    for i in range(num_episodes):
        obs = env.reset()[0]
        state = obs['observation']  # Extract observation from the info dict
        ep_reward = 0
        step = 0
        while True:
            step += 1
            if gif_file is not None and env.render_mode == 'rgb_array':
                frame = env.render()
                frames.append(_label_with_episode_number(frame, i, step))
            action = agent.choose_action(state) 
            next_obs, reward, done, _, _ = env.step(action) 
            next_state = next_obs['observation']
            state = next_state
            ep_reward += reward
            if max_steps is not None and step > max_steps:
                done = True
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

###############

if __name__ == "__main__":

    # check if GPU is available
    if not tf.test.is_gpu_available():
        print("No GPU found, Exiting.")
        exit(1)
    # Create the LunarLanderContinuous-v3 environment
    env = gym.make('FetchReachDense-v3', max_episode_steps=100, render_mode='rgb_array')

    obs_shape = env.observation_space['observation'].shape
    action_shape = env.action_space.shape
    print(f"Observation shape: {obs_shape}, Action shape: {action_shape}")
    action_upper_bound = env.action_space.high  # Assuming continuous action space
    print(f"Action upper bound: {action_upper_bound}")
    print(f"Action lower bound: {env.action_space.low}")

    if EXT_MODEL:
        # Create the actor and critic models
        actor_net = create_actor_model(obs_shape, action_shape[0])
        critic_net = create_critic_model(obs_shape, action_shape)
        value_net = create_value_network(obs_shape)
    else:
        actor_net = None
        critic_net = None
        value_net = None

    if SAC2:
        # Initialize the SAC agent
        agent = SACAgent(obs_shape, action_shape,
                        action_upper_bound=action_upper_bound,
                        reward_scale=5.0,
                        buffer_size=1000000,
                        batch_size=256,
                        actor_model=actor_net,
                        critic_model=critic_net,
                        value_model=value_net,
                        max_grad_norm=None,)
    else:
        # Initialize the SAC agent
        agent = SACAgent(obs_shape, action_shape,
                        action_upper_bound=action_upper_bound,
                        reward_scale=5.0,
                        buffer_size=1000000,
                        batch_size=256,
                        actor_model=actor_net,
                        critic_model=critic_net,
                        max_grad_norm=None,)

    if not EVAL:
        # train the agent
        train_sac_agent(env, agent, num_episodes=1500,
                        max_score=None, min_score=None,
                        stop_score=0,
                        ep_max_steps=None,
                        update_per_step=1,
                        wandb_log=True)
    else:
        # load the weights
        agent.load_weights(filename='fr_sac.weights.h5')
        # validate the agent
        validate(env, agent, num_episodes=10, max_steps=1000, gif_file='fr_sac.gif')