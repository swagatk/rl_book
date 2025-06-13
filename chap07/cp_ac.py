'''
Solving CartPole problem using Actor Critic Algorithm
'''
import gymnasium as gym
from actor_critic import ACAgent 
import wandb 
import os
import numpy as np

CFG = dict(
    lr_a = 1e-4,
    lr_c = 1e-4, 
    gamma = 0.99,
    agent = 'ac',
)

def ac_train(env, agent, max_episodes=10000, log_freq=50, 
             stop_score=200, filename=None, wandb_log=False):
    print('Environment name: ', env.spec.id)
    print('RL Agent name:', agent.name)
    
    assert isinstance(env.action_space, gym.spaces.Discrete),\
                "AC Agent only for discrete action spaces"

    if filename is not None:
        file = open(filename, 'w')

    if wandb_log:
        run = wandb.init(entity='swagatk', project=env.spec.id, config=CFG)

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
        # while loop ends here
        ep_scores.append(ep_score)
        if filename is not None:
            file.write(f'{e}\t{ep_score}\t{np.mean(ep_scores)}\t{a_loss}\t{c_loss}\n')
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
                'actor_loss': np.mean(a_losses),
                'critic_loss': np.mean(c_losses),
                'mean_score': np.mean(ep_scores),
                'best_score': best_score,
            })

        if ep_score > best_score:
            best_score = ep_score
            agent.save_weights()
            print(f'Best Score: {ep_score}, episode: {e}. Model saved.')

        if np.mean(ep_scores[-100:]) > stop_score:
            print('The problem is solved in {} episodes'.format(e))
            break
    # for loop ends here
    if filename is not None:
        file.close()
    if wandb.log:
        run.finish()

if __name__ == '__main__':
    env = gym.make('CartPole-v1')
    obs_shape = env.observation_space.shape
    action_size = env.action_space.n

    print("Observation shape: ", obs_shape)
    print("Action Size: ", action_size)
    print("Max Episode steps: ", env.spec.max_episode_steps)

    # create an RL agent
    agent = ACAgent(obs_shape, action_size)

    # train the RL agent on
    ac_train(env, agent, max_episodes=1500, log_freq=100, 
             stop_score=499, wandb_log=True)
