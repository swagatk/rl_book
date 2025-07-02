'''
Asynchronous Advantage Actor-Critic (A3C) implementation using TensorFlow and Gymnasium.

* It uses A2CAgent class from a2c.py to implement the A3C algorithm.
* In this version, discounted returns are computed for each episode. Next_states and dones are not collected.  
* It uses multipprocessing to run multiple workers in parallel, each worker interacts with the environment and updates a global network.
* Gradients are collected from each worker and averaged before updating the global network.
* It runs on multi-core CPU and does not use GPU for training.
* Trying to make this code general enough to work with any environment. 
'''
import gymnasium as gym
import numpy as np
import multiprocessing
import time
import wandb
import os
import gc
import queue
from a2c import A2CAgent
from collections import deque
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Disable GPU for this script

import tensorflow as tf
import tensorflow_probability as tfp

# Hyperparameters
DEBUG = False  # Set to True for debugging output


# Worker function for each process
def worker(worker_id, global_weights_queue, 
           gradients_queue,
           save_request_queue,
           env_id, 
           create_actor_func=None,
           create_critic_func=None,
           max_episodes=1500,
           max_score = 500, 
           min_score = -500, 
           max_steps = None, 
           wandb_log=False):

    # Set random seed for reproducibility in each process
    tf.random.set_seed(worker_id + 1)
    np.random.seed(worker_id + 1)

    # create environment
    env = gym.make(env_id)
    obs_shape = env.observation_space.shape
    action_size = env.action_space.n

    # Create local network and environment
    if create_actor_func is not None:
        actor = create_actor_func(obs_shape, action_size)
    else:
        actor = None

    if create_critic_func is not None:
        critic = create_critic_func(obs_shape)
    else:
        critic = None
    local_network = A2CAgent(obs_shape, action_size,
                             a_model=actor, c_model=critic)

    if wandb_log and worker_id == 0:
        run = wandb.init(
            project=env.spec.id,  # Replace with your WandB project name
            entity='swagatk',  # Replace with your WandB entity
            config={
                'lr_a': local_network.lr_a,
                'lr_c': local_network.lr_c,
                'gamma': local_network.gamma,
                'group': 'expt_11',
                'agent': 'A3C'
            }

        )

    episode = 0
    ep_scores = [] 
    best_score = -np.inf
    for episode in range(max_episodes):
        weights_updated = False
        for _ in range(3):  # Retry up to 3 times
            try:
                global_weights = global_weights_queue.get_nowait()
                local_network.set_weights(*global_weights)
                weights_updated = True
                if DEBUG:
                    print(f"Worker {worker_id}: Retrieved global weights from queue")
                break
            except queue.Empty:
                if DEBUG:
                    print(f"Worker {worker_id}: Global weights queue empty, retrying...")
                time.sleep(0.1)  # Brief pause before retry
        if not weights_updated:
            if DEBUG:
                print(f"Worker {worker_id}: Using current weights after failed attempts")
            # Continue with current weights if queue is empty

        state = env.reset()[0]
        episode_reward = 0
        done = False
        step = 0
        states = deque(maxlen=max_steps if max_steps is not None else 1000) 
        actions = deque(maxlen=max_steps if max_steps is not None else 1000) 
        rewards = deque(maxlen=max_steps if max_steps is not None else 1000)

        # Collect trajectory
        while not done: 
            action = int(local_network.policy(state))
            next_state, reward, done, truncated, _ = env.step(action)
            done = done or truncated

            states.append(state) # type: ignore
            actions.append(action)
            rewards.append(reward)

            state = next_state
            episode_reward += reward

            step += 1

            if max_score is not None and episode_reward >= max_score:
                done = True
            if min_score is not None and episode_reward <= min_score:
                done = True
            if max_steps is not None and step >= max_steps:
                done = True

        # end of episode
        ep_scores.append(episode_reward)

        # update the local network 
        a_loss, c_loss, actor_grads, critic_grads = local_network.compute_gradients(
            states, actions, rewards 
        )

        # Update global network
        try:
            gradients_queue.put_nowait((actor_grads, critic_grads))
        except queue.Full: 
            if DEBUG:
                print(f"Worker {worker_id}: Gradients queue is full, skipping update.")
            continue


        if episode_reward > best_score:
            best_score = episode_reward
        if wandb_log and worker_id == 0:
            wandb.log({
                'episode': episode,
                'ep_score': episode_reward, 
                'avg100score': np.mean(ep_scores[-100:]),
                'mean_score': np.mean(ep_scores),
                'best_score': best_score,
                'actor_loss': a_loss,
                'critic_loss': c_loss,
            })

        try:
            save_request_queue.put_nowait((worker_id, episode, episode_reward))
        except queue.Full:
            if DEBUG:
                print(f"Worker {worker_id}: Save request queue is full, skipping save request.")
            continue

        # free memory  
        tf.keras.backend.clear_session()  # Clear TensorFlow session
        states.clear()
        actions.clear()
        rewards.clear()
        gc.collect()  # Collect garbage to free memory
    # end of for-loop
    env.close()
    if wandb_log and worker_id == 0:
        run.finish()

def run_workers(env_name, max_num_workers=5, max_episodes=1500, 
         wandb_log=True, max_score=500, min_score=-200,
         max_steps=1000,
         create_actor_func=None,
         create_critic_func=None):
    # Set random seed for reproducibility
    tf.random.set_seed(42)
    np.random.seed(42)

    N_WORKERS = min(multiprocessing.cpu_count(), max_num_workers)
    print('N_WORKERS:', N_WORKERS)


    # Initialize environment to get state and action sizes
    env = gym.make(env_name)

    # check if the environment has a discrete action space
    if not isinstance(env.action_space, gym.spaces.Discrete):
        raise ValueError("A3C Agent currently supports discrete action spaces only")

    obs_shape = env.observation_space.shape
    action_size = env.action_space.n
    env_id = env.spec.id
    env.close()

    if create_actor_func is not None: 
        a_model = create_actor_func(obs_shape, action_size)
    else:
        a_model = None
    if create_critic_func is not None:
        c_model = create_critic_func(obs_shape)
    else:
        c_model = None

    # Initialize global network
    global_network = A2CAgent(obs_shape, action_size,
                              a_model=a_model, c_model=c_model)

    # Create a queue to share weights between processes
    manager = multiprocessing.Manager()
    global_weights_queue = manager.Queue(maxsize=2*N_WORKERS)
    gradients_queue = manager.Queue(maxsize=2*N_WORKERS)
    save_request_queue = manager.Queue(maxsize=N_WORKERS)
    save_lock = manager.Lock()

    # initial weights for the global network
    for _ in range(2*N_WORKERS):
        # Initialize the global weights queue with the global network's weights
        global_weights_queue.put(global_network.get_weights())


    # Create and start worker processes
    processes = []
    for i in range(N_WORKERS):
        p = multiprocessing.Process(
            target=worker,
            args=(i, global_weights_queue, gradients_queue, 
                  save_request_queue, env_id, 
                  create_actor_func, create_critic_func,  
                  max_episodes,
                  max_score, min_score, max_steps, wandb_log)
        )
        p.start()
        processes.append(p)
        print(f'Started worker {p.name}')

    try:
        last_refill_time = time.time()
        best_reward = -np.inf
        while(any(p.is_alive() for p in processes)):
            gradients = []
            for _ in range(N_WORKERS):
                if not gradients_queue.empty():
                    gradients.append(gradients_queue.get())

            if gradients: # process gradients if available 
                # Filter out None gradients
                valid_gradients = [
                    (actor_grads, critic_grads)
                    for actor_grads, critic_grads in gradients
                    if actor_grads is not None and critic_grads is not None
                ]

                if valid_gradients:  # Only proceed if there are valid gradients
                    actor_grads_avg = []
                    critic_grads_avg = []
                    for actor_grads, critic_grads in valid_gradients:
                        if not actor_grads_avg:
                            actor_grads_avg = [tf.convert_to_tensor(g) for g in actor_grads]
                            critic_grads_avg = [tf.convert_to_tensor(g) for g in critic_grads]
                        else:
                            for i, g in enumerate(actor_grads):
                                actor_grads_avg[i] = tf.add(actor_grads_avg[i], tf.convert_to_tensor(g))
                            for i, g in enumerate(critic_grads):
                                critic_grads_avg[i] = tf.add(critic_grads_avg[i], tf.convert_to_tensor(g))

                    n = len(valid_gradients)
                    actor_grads_avg = [tf.math.truediv(g, n) for g in actor_grads_avg]
                    critic_grads_avg = [tf.math.truediv(g, n) for g in critic_grads_avg]

                    global_network.apply_gradients(actor_grads_avg, critic_grads_avg)
            
                # Synchronize global weights with all workers
                updated_weights = global_network.get_weights()
                for _ in range(N_WORKERS):
                    try:
                        # Synchronize global weights with all workers
                        global_weights_queue.put_nowait(updated_weights)
                    except queue.Full:
                        if DEBUG:
                            print("Global weights queue is full, skipping synchronization.")
                        while not global_weights_queue.empty():
                            try: 
                                global_weights_queue.get_nowait()
                            except queue.Empty:
                                break
                        global_weights_queue.put_nowait(updated_weights)

            # periodically refill the global weights queue
            if time.time() - last_refill_time > 5:
                try:
                    global_weights_queue.put_nowait(global_network.get_weights())
                    last_refill_time = time.time()
                except queue.Full:
                    if DEBUG:
                        print("Global weights queue is full, skipping refill.")

            # saving weights
            while not save_request_queue.empty():
                try:
                    worker_id, episode, episode_reward = save_request_queue.get_nowait()
                    print(f"Worker: {worker_id}, episode: {episode}, reward: {episode_reward:.2f}")
                    if episode_reward > best_reward:
                        best_reward = episode_reward
                        with save_lock:
                            try:
                                global_network.save_weights(
                                    actor_wt_file='a3c_actor.weights.h5',
                                    critic_wt_file='a3c_critic.weights.h5',
                                )
                                print(f"Best Score: {best_reward}. Saved weights!")
                            except Exception as e:
                                print(f"Error saving weights: {e}")
                except queue.Empty:
                    break
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        print("Shutting down workers...")
        #Wait for all processes to complete
        for p in processes:
            p.join()
            print(f'Worker {p.name} has finished.')



    
