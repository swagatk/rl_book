import collections
import numpy as np
import gymnasium as gym
import wandb
from typing import cast

from dreamerv1 import DreamerV1


def make_env():
    env = gym.make("Reacher-v4")
    return env


def train():
    config = {
        "env_id": "Reacher-v4",
        "h_size": 256,
        "z_size": 32,
        "batch_size": 32,
        "seq_len": 50,
        "horizon": 15,
        "kl_scale": 0.1,
        "hidden_nodes": 256,
        "gamma": 0.99,
        "lambda": 0.95,
        "lr_model": 3e-4,
        "lr_agent": 5e-5,
        "max_episodes": 800,
        "prefill_steps": 5000,
        "train_every": 1,
        "train_updates": 1,
        "actor_noise_init": 0.35,
        "actor_noise_min": 0.05,
        "actor_noise_decay": 80000,
        "target_critic_tau": 0.01,
        "bufferlen": 100000,
        "log_freq": 10,
    }

    wandb.init(project="dreamerv1-reacher-v4", config=config)

    env = make_env()
    obs_shape = env.observation_space.shape
    action_dim = cast(gym.spaces.Box, env.action_space).shape[0]

    agent = DreamerV1(
        obs_shape=obs_shape,
        action_dim=action_dim,
        h_size=config["h_size"],
        z_size=config["z_size"],
        batch_size=config["batch_size"],
        seq_len=config["seq_len"],
        horizon=config["horizon"],
        kl_scale=config["kl_scale"],
        hidden_nodes=config["hidden_nodes"],
        lr_model=config["lr_model"],
        lr_agent=config["lr_agent"],
        gamma=config["gamma"],
        lambda_=config["lambda"],
        actor_noise_init=config["actor_noise_init"],
        actor_noise_min=config["actor_noise_min"],
        actor_noise_decay=config["actor_noise_decay"],
        target_critic_tau=config["target_critic_tau"],
        bufferlen=config["bufferlen"],
    )

    reward_history = collections.deque(maxlen=100)
    global_step = 0

    for episode in range(config["max_episodes"]):
        obs, _ = env.reset()
        done = False
        episode_reward = 0.0
        episode_steps = 0

        state = agent.rssm.get_initial_state(1)
        prev_action = np.zeros(action_dim, dtype=np.float32)

        while not done:
            if global_step < config["prefill_steps"]:
                # Keep latent state synchronized during random exploration.
                _, state = agent.select_action(obs, state, prev_action=prev_action, training=False)
                action = env.action_space.sample()
            else:
                action, state = agent.select_action(obs, state, prev_action=prev_action, training=True)
                action = action.numpy()

            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # Replay stores post-action observation for RSSM training alignment.
            agent.buffer.append((next_obs, action, reward, done))

            if global_step >= config["prefill_steps"] and global_step % config["train_every"] == 0:
                model_loss, agent_loss = None, None
                for _ in range(config["train_updates"]):
                    model_loss, agent_loss = agent.train()
                if model_loss is not None and agent_loss is not None:
                    wandb.log(
                        {
                            "model_loss": float(model_loss),
                            "agent_loss": float(agent_loss),
                            "global_step": global_step,
                        }
                    )

            obs = next_obs
            prev_action = action
            episode_reward += float(reward)
            episode_steps += 1
            global_step += 1

        reward_history.append(episode_reward)
        avg_reward = float(np.mean(reward_history))

        if episode % config["log_freq"] == 0:
            print(f"Episode: {episode}, Reward: {episode_reward:.2f}, Avg100: {avg_reward:.2f}, Steps: {episode_steps}")

        wandb.log(
            {
                "episode": episode,
                "episode_reward": episode_reward,
                "avg_ep_reward": avg_reward,
                "episode_steps": episode_steps,
                "global_step": global_step,
            }
        )

    env.close()
    wandb.finish()


if __name__ == "__main__":
    train()
