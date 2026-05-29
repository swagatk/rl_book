from dreamerv1 import DreamerV1 
import wandb
import gymnasium as gym
import tensorflow as tf
import numpy as np
import collections

# --- Configuration (Optimized for LunarLander) ---
H_SIZE = 256
Z_SIZE = 32
BATCH_SIZE = 16
SEQ_LEN = 50        # Increased for better temporal dependencies
HORIZON = 15        # Slightly longer imagination horizon
LR_MODEL = 3e-4
LR_AGENT = 1e-4
KL_SCALE = 0.1      # KL Balancing scale
MAX_HIDDEN_NODES = 256
MAX_BUFFER_LEN = 100000


# --- Training Loop for LunarLander ---
env = gym.make("LunarLanderContinuous-v3")
agent = DreamerV1(env.observation_space.shape, env.action_space.shape[0],
                  h_size=H_SIZE, 
                  z_size=Z_SIZE, 
                  batch_size=BATCH_SIZE, 
                  seq_len=SEQ_LEN, 
                  horizon=HORIZON, 
                  kl_scale=KL_SCALE,
                  lr_model=LR_MODEL, 
                  lr_agent=LR_AGENT,
                  bufferlen=MAX_BUFFER_LEN,
                  hidden_nodes=MAX_HIDDEN_NODES)

wandb.init(project="dreamer-v1-lunar-lander", config={"h_size": H_SIZE, "z_size": Z_SIZE, "seq_len": SEQ_LEN})

reward_history = collections.deque(maxlen=100)
total_steps = 0
SEED_STEPS = 1000  # Initial random exploration

for episode in range(1, 301):
    obs, _ = env.reset()
    state = agent.rssm.get_initial_state(1)
    episode_reward = 0

    for step in range(500):
        if total_steps < SEED_STEPS:
            action = env.action_space.sample()
            # Update state with zero-action placeholder to keep RSSM aligned
            _, state = agent.select_action(obs, state, training=True)
        else:
            action, state = agent.select_action(obs, state)
            action = action.numpy()

        next_obs, reward, done, trunc, _ = env.step(action)
        agent.buffer.append((obs, action, reward))

        obs = next_obs
        episode_reward += reward
        total_steps += 1

        if total_steps > SEED_STEPS and total_steps % 10 == 0:
            model_loss, agent_loss = agent.train()
            wandb.log({"model_loss": model_loss, "agent_loss": agent_loss, "global_step": total_steps})

        if done or trunc:
            break

    reward_history.append(episode_reward)
    avg_reward = np.mean(reward_history)
    wandb.log({"episode": episode, "reward": episode_reward, "avg_reward": avg_reward})

    if episode % 10 == 0:
        print(f"Episode {episode} | Avg Reward: {avg_reward:.2f}")

wandb.finish()