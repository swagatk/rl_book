import argparse
import collections
import numpy as np
import tensorflow as tf
import gymnasium as gym
import shimmy  # Registers dm_control/* environments with Gymnasium.
import wandb
import os

from dreamerv1 import DreamerV1

def make_env():
    env = gym.make("dm_control/cheetah-run-v0")
    env = gym.wrappers.FlattenObservation(env)
    return env

def train():
    wandb.init(project="dreamerv1-cheetah-run-v0", config={
        "obs_shape": (17,),  # Flattened observation shape for cheetah-run-v0
        "action_dim": 6,     # Action dimension for cheetah-run-v0
        "h_size": 256,
        "z_size": 32,
        "batch_size": 16,
        "seq_len": 30,
        "horizon": 10,
        "kl_scale": 0.1,
        "hidden_nodes": 256,
        "max_episodes": 500,
        "prefill_steps": 3000,
        "train_every": 10,
        "train_updates": 2,
        "gamma": 0.99,
        "lambda": 0.95
    })
    
    env = make_env()
    obs_shape = env.observation_space.shape
    action_dim = env.action_space.shape[0]
    
    agent = DreamerV1(
        obs_shape=obs_shape,
        action_dim=action_dim,
        h_size=256, 
        z_size=32,
        batch_size=16,
        seq_len=30,
        horizon=10,
        kl_scale=0.1,
        hidden_nodes=256,
        gamma=0.99,
        lambda_=0.95
    )
    
    max_episodes = 500
    prefill_steps = 3000
    train_every = 10
    train_updates = 2
    global_step = 0
    log_freq = 10
    
    ep_rewards = []
    for ep in range(max_episodes):
        obs, _ = env.reset()
        done = False
        episode_reward = 0
        ep_step = 0
        
        # Initial state for RSSM
        state = agent.rssm.get_initial_state(1)
        prev_action = np.zeros(action_dim, dtype=np.float32)
        
        while not done:
            if global_step < prefill_steps:
                action = env.action_space.sample()
            else:
                action, state = agent.select_action(obs, state, prev_action=prev_action, training=True)
                action = action.numpy()
                
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # Store next_obs so RSSM posterior at t is conditioned on the observation after action_t.
            agent.buffer.append((next_obs, action, reward, done))
            
            if global_step >= prefill_steps and global_step % train_every == 0:
                for _ in range(train_updates):
                    model_loss, agent_loss = agent.train()
                if model_loss is not None:
                    wandb.log({"model_loss": model_loss, "agent_loss": agent_loss, "global_step": global_step})
            
            obs = next_obs
            prev_action = action
            episode_reward += reward
            global_step += 1
            ep_step += 1

        # end of episode, log reward
        ep_rewards.append(episode_reward)
        if ep % log_freq == 0:
            print(f"Episode: {ep}, Reward: {episode_reward:.2f}, Steps: {ep_step}")
        avg_ep_reward = np.mean(ep_rewards[-100:])

        wandb.log({"episode_reward": episode_reward, "global_step": global_step, "episode": ep,
                   "avg_ep_reward": avg_ep_reward})

if __name__ == "__main__":
    train()
