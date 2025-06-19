"""
A3C (Asynchronous Advantage Actor-Critic) implementation using TensorFlow and Gymnasium.
Basic version with multiple workers for parallel training on the LunarLander-v3 environment.

It uses multi-core cpu processing to run multiple workers in parallel, each worker interacts with the environment and updates a global network.
The global network is updated asynchronously by the workers, which helps in stabilizing the training process.
This implementation includes logging with Weights & Biases (wandb) for tracking training progress.
"""
import gymnasium as gym
import numpy as np
import multiprocessing
import time
import wandb
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Disable GPU for this script

import tensorflow as tf
from tensorflow.keras import layers, Model
import tensorflow_probability as tfp

# Hyperparameters
GLOBAL_MAX_EPISODES = 2000
T_MAX = 1000 
GAMMA = 0.99
LR_ACTOR = 0.0001
LR_CRITIC = 0.001
N_WORKERS = 20 # multiprocessing.cpu_count()
ENTROPY_BETA = 0.01


# Global network definition
class A3CNetwork(Model):
    def __init__(self, state_size, action_size):
        super(A3CNetwork, self).__init__()
        self.actor = tf.keras.Sequential([
            layers.Input(shape=(state_size,)),
            layers.Dense(128, activation='relu'),
            layers.Dense(64, activation='relu'),
            layers.Dense(action_size, activation='softmax')
        ])
        self.critic = tf.keras.Sequential([
            layers.Input(shape=(state_size,)),
            layers.Dense(128, activation='relu'),
            layers.Dense(64, activation='relu'),
            layers.Dense(1)
        ])

    def call(self, state): 
        policy = self.actor(state) # returns action probabilities
        value = self.critic(state) # returns state value
        return policy, value

    def set_weights(self, weights):
        """Set the weights of the actor and critic networks."""
        self.actor.set_weights(weights[0])
        self.critic.set_weights(weights[1])
    
    def get_weights(self):
        """Get the weights of the actor and critic networks."""
        return [self.actor.get_weights(), self.critic.get_weights()]


# Worker function for each process
def worker(worker_id, global_weights_queue, 
           state_size, action_size, 
           max_score = 300, 
           min_score = -300, 
           wandb_log=False):

    # Set random seed for reproducibility in each process
    tf.random.set_seed(worker_id + 1)
    np.random.seed(worker_id + 1)

    # Create local network and environment
    local_network = A3CNetwork(state_size, action_size)
    optimizer_actor = tf.keras.optimizers.Adam(learning_rate=LR_ACTOR)
    optimizer_critic = tf.keras.optimizers.Adam(learning_rate=LR_CRITIC)
    env = gym.make('LunarLander-v3')

    if wandb_log and worker_id == 0:
        run = wandb.init(
            project=env.spec.id,  # Replace with your WandB project name
            entity='swagatk',  # Replace with your WandB entity
            config={
                'worker_id': worker_id,
                'lr_a': LR_ACTOR,
                'lr_c': LR_CRITIC,
                'gamma': GAMMA,
                'group': 'expt_9',
                'agent': 'A3C_V1',
            }
        )

    episode = 0
    ep_scores = []
    best_score = -np.inf
    while episode < GLOBAL_MAX_EPISODES:
        # Synchronize local network with global weights
        try:
            global_weights = global_weights_queue.get_nowait()
            local_network.set_weights(global_weights)
        except:
            pass

        state = env.reset()[0]
        episode_reward = 0
        done = False
        step = 0
        states, actions, rewards = [], [], []

        # Collect trajectory
        while not done and step < T_MAX:
            state = np.reshape(state, [1, state_size])
            policy, value = local_network(state)
            action = np.random.choice(action_size, p=policy.numpy()[0])
            next_state, reward, done, truncated, _ = env.step(action)
            done = done or truncated

            states.append(state)
            actions.append(action)
            rewards.append(reward)

            state = next_state
            episode_reward += reward

            if max_score is not None and episode_reward >= max_score:
                done = True
            if min_score is not None and episode_reward <= min_score:
                done = True
            step += 1

        # end of episode
        # Update episode count
        episode += 1
        ep_scores.append(episode_reward)
        #rewards_queue.put(episode_reward)
        # Print progress
        print(f"Worker {worker_id}, Episode {episode}, Reward: {episode_reward:.2f}")

        if episode_reward > best_score:
            best_score = episode_reward
        if wandb_log and worker_id == 0:
            wandb.log({
                'episode': episode,
                'ep_score': episode_reward, 
                'avg100score': np.mean(ep_scores[-100:]),
                'mean_score': np.mean(ep_scores),
                'best_score': best_score,
            })

        # Compute returns and advantages
        states = np.vstack(states)
        actions = np.array(actions)
        rewards = np.array(rewards)
        next_state = np.reshape(next_state, [1, state_size])
        _, next_value = local_network(next_state)
        next_value = next_value.numpy()[0, 0] if not done else 0.0

        returns = []
        R = next_value
        for r in rewards[::-1]:
            R = r + GAMMA * R
            returns.insert(0, R)
        returns = np.array(returns)
        advantages = returns - local_network(states)[1].numpy().flatten()

        # Compute gradients
        with tf.GradientTape() as tape_actor, tf.GradientTape() as tape_critic:
            policy, value = local_network(states)
            action_indices = tf.one_hot(actions, action_size)
            action_probs = tf.reduce_sum(policy * action_indices, axis=1)
            log_probs = tf.math.log(action_probs + 1e-10)
            actor_loss = -tf.reduce_mean(log_probs * advantages)
            entropy = -tf.reduce_mean(tf.reduce_sum(policy * tf.math.log(policy + 1e-10), axis=1))
            actor_loss = actor_loss - ENTROPY_BETA * entropy
            critic_loss = tf.reduce_mean(tf.square(returns - value))

        actor_grads = tape_actor.gradient(actor_loss, local_network.actor.trainable_variables)
        critic_grads = tape_critic.gradient(critic_loss, local_network.critic.trainable_variables)

        # Apply gradients to the local network
        optimizer_actor.apply_gradients(zip(actor_grads, local_network.actor.trainable_variables))
        optimizer_critic.apply_gradients(zip(critic_grads, local_network.critic.trainable_variables))

        # Update global network
        local_weights = local_network.get_weights()
        global_weights_queue.put(local_weights)

    # end of while-loop
    env.close()
    if wandb_log and worker_id == 0:
        run.finish()

def main(wandb_log=True, max_score=500, min_score=-500):
    # Set random seed for reproducibility
    tf.random.set_seed(42)
    np.random.seed(42)

    print('N_WORKERS:', N_WORKERS)

    # Initialize environment to get state and action sizes
    env = gym.make('LunarLander-v3')
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    env.close()

    # Initialize global network
    global_network = A3CNetwork(state_size, action_size)

    # Create a queue to share weights between processes
    manager = multiprocessing.Manager()
    global_weights_queue = manager.Queue()
    global_weights_queue.put(global_network.get_weights())

    # Create and start worker processes
    processes = []
    for i in range(N_WORKERS):
        p = multiprocessing.Process(
            target=worker,
            args=(i, global_weights_queue, state_size, action_size, 
                  max_score, min_score, wandb_log)
        )
        p.start()
        processes.append(p)
        print(f'Started worker {p.name}')

    # Wait for all processes to complete
    for p in processes:
        p.join()
        print(f'Worker {p.name} has finished.')

if __name__ == '__main__':
    multiprocessing.set_start_method('spawn')
    main()