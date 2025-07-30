"""
Solving the LunarLanderContinuous-v3 environment using Soft Actor-Critic (SAC) algorithm.
"""
import gymnasium as gym
import numpy as np
import wandb
import os
import sys
import tensorflow as tf
from sac2 import SACAgent

SAC2 = True

def train_sac_agent(env, agent, num_episodes=1500, 
                    max_score=500, min_score=-300, 
                    stop_score=200, 
                    warmup_steps=1000,
                    ep_max_steps=1000,
                    update_per_step=1,
                    log_freq=100,
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
    if SAC2:
        v_loss = 0
    for e in range(num_episodes):
        done = False
        truncated = False
        state = env.reset()[0]
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
            next_state, reward, done, truncated, _ = env.step(action)
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
            if total_steps >= warmup_steps:
                if SAC2:
                    vl, cl, al, alphal = agent.train()   
                else:
                   cl, al, alphal = agent.train(update_per_step=update_per_step)   
                c_losses.append(cl)
                a_losses.append(al)
                alpha_losses.append(alphal)            
                if SAC2:
                    v_losses.append(vl)


        # while loop ends here
        ep_scores.append(ep_score)
        c_loss = np.mean(c_losses) 
        a_loss = np.mean(a_losses) 
        alpha_loss = np.mean(alpha_losses)
        if SAC2:
            v_loss = np.mean(v_losses)


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
                'v_loss': v_loss if SAC2 else None
            })

        if ep_score > best_score:
            best_score = ep_score
            agent.save_weights(filename='ll_sac.weights.h5')
            print(f'Best Score: {ep_score}, episode: {e}. Model saved.')

        if np.mean(ep_scores[-100:]) > stop_score:
            print('The problem is solved in {} episodes'.format(e))
            break
    # for loop ends here
    if filename is not None:
        file.close()
    if wandb_log:
        run.finish()


if __name__ == "__main__":

    # check if GPU is available
    if not tf.test.is_gpu_available():
        print("No GPU found, Exiting.")
        exit(1)
    # Create the LunarLanderContinuous-v3 environment
    env = gym.make('LunarLander-v3', continuous=True)

    obs_shape = env.observation_space.shape
    action_shape = env.action_space.shape
    print(f"Observation shape: {obs_shape}, Action shape: {action_shape}")
    action_upper_bound = env.action_space.high  # Assuming continuous action space
    print(f"Action upper bound: {action_upper_bound}")
    print(f"Action lower bound: {env.action_space.low}")

    # Initialize the SAC agent
    agent = SACAgent(obs_shape, action_shape,
                    buffer_size=1000000,
                    batch_size=256,
                     action_upper_bound=action_upper_bound,
                     reward_scale=1.0)

    # train the agent
    train_sac_agent(env, agent, num_episodes=1500,
                    max_score=500, min_score=-300,
                    stop_score=200,
                    update_per_step=1,
                    wandb_log=True)