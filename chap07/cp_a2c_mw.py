'''
A2C Multi-Worker Agent for Discrete Action Spaces
'''

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from multiprocessing import Process, Queue, Barrier, Lock 
import wandb
from a2c_sw import A2CAgent
import gymnasium as gym


def preprocess1(states, actions, rewards, ep_scores, gamma, s_queue, a_queue, r_queue, ep_score_queue, lock):
    """
    Preprocesses the states, actions, and rewards for A2C training.
    This function is run in a separate process.
    """
    discounted_rewards = []
    sum_reward = 0
    for r in reversed(rewards):
        sum_reward = r + gamma * sum_reward
        discounted_rewards.insert(0, sum_reward)
    
    states = np.array(states, dtype=np.float32)
    actions = np.array(actions, dtype=np.int32)
    discounted_rewards = np.array(discounted_rewards, dtype=np.float32)
    ep_scores = np.array(ep_scores, dtype=np.float32)

    lock.acquire()
    s_queue.put(states)
    a_queue.put(actions)
    r_queue.put(discounted_rewards)
    ep_score_queue.put(ep_scores)

    lock.release()

def preprocess2(s_queue, a_queue, r_queue, ep_scores_queue):
    """
    Collects preprocessed states, actions, and rewards from the queues.
    This function is run in a separate process.
    """
    states = []
    while not s_queue.empty():
        states.append(s_queue.get())
    actions = []
    while not a_queue.empty():
        actions.append(a_queue.get())
    dis_rewards = []
    while not r_queue.empty():
        dis_rewards.append(r_queue.get())

    ep_scores = []
    while not ep_score_queue.empty():
        ep_scores.append(ep_score_queue.get())

    state_batch = np.concatenate(*(states,), axis=0)
    action_batch = np.concatenate(*(actions,), axis=None)
    reward_batch = np.concatenate(*(dis_rewards,), axis=None)
    ep_score_batch = np.concatenate(*(ep_scores,), axis=None)

    return state_batch, action_batch, reward_batch, ep_score_batch


def runner(barrier, lock, env, s_queue, a_queue, r_queue, ep_scores, max_qsize=10, max_episodes=2000, log_freq=100, wandb_log=False):
    tf.random.set_seed(42)
    obs_shape = env.observation_space.shape
    action_size = env.action_space.n 
    agent = A2CAgent(obs_shape, action_size)
    ep_scores = []
    for e in range(max_episodes):
        state = env.reset()[0]
        done = False
        states, actions, rewards = [], [], []
        ep_reward = 0
        all_a_loss, all_c_loss = [], []

        while not done:
            action = agent.policy(state)

            next_state, reward, done, _, _ = env.step(action)
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            state = next_state
            ep_reward += reward

        ep_scores.append(ep_reward)
        if e  % log_freq == 0:
            print(f'Worker {barrier.parties - 1} Episode {e}, Reward: {ep_reward:.2f}, Avg Reward: {np.mean(ep_scores):.2f} \
                  Avg100 Reward: {np.mean(ep_scores[-100:]):.2f}')
            

        # Preprocess the collected data
        preprocess1(states, actions, rewards, ep_scores, agent.gamma, s_queue, a_queue, r_queue, ep_score_queue, lock)

        # Synchronize with other workers
        b = barrier.wait()
        if b == 0: # all workers have reached this point
            if (s_queue.qsize() == max_qsize and 
                a_queue.qsize() == max_qsize and 
                r_queue.qsize() == max_qsize):

                state_batch, action_batch, reward_batch, ep_score_batch = preprocess2(s_queue, a_queue, r_queue, ep_score_queue)

                # train the agent
                a_loss, c_loss = agent.train(state_batch, action_batch, reward_batch)
                all_a_loss.append(a_loss)
                all_c_loss.append(c_loss)

            if wandb_log:
                wandb.log({
                    'episode': e,
                    'ep_reward': ep_reward, 
                    'avg_reward': np.mean(ep_score_batch),
                    'avg100_reward': np.mean(ep_score_batch[-100:]),
                    'actor_loss': np.mean(all_a_loss),
                    'critic_loss': np.mean(all_c_loss),
                })

        barrier.wait()


if __name__ == "__main__":

    if not tf.test.is_gpu_available():
        print("No GPU available. Exiting.")
        exit(0)
    env = gym.make('CartPole-v1')

    

    run = wandb.init(entity='swagatk', project=env.spec.id, 
        config={
            'lr_a': 1e-3,
            'lr_c': 1e-3,
            'gamma': 0.99,
            'agent': 'a2c_mw'})

    barrier = Barrier(10)  # 10 workers
    s_queue = Queue(maxsize=10)
    a_queue = Queue(maxsize=10)
    r_queue = Queue(maxsize=10)
    ep_score_queue = Queue(maxsize=10)

    lock = Lock()
    processes = []
    for i in range(barrier.parties):
        worker = Process(target=runner, args=(barrier, lock, env, s_queue, a_queue, r_queue, ep_scores
                                              max_qsize=10, max_episodes=2000, wandb_log=True))
        processes.append(worker)
        worker.start()

    for process in processes:
        process.join()
    env.close()
    print("Training completed.")
    run.finish()


    