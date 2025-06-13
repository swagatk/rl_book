'''
REINFORCE agent for CartPole-v0
'''
import os
import gymnasium as gym
import numpy as np
from reinforce import REINFORCEAgent 
from utils import plot_datafile
import wandb
from utils import validate
########################

# training an agent for a problem environment
def train(env, agent, max_episodes=1000, filename=None, 
                        ep_max_score=None, ep_min_score=None,
                        stop_score=200, log_freq=100,
                        wandb_log=False):
    if wandb_log:
        run = wandb.init(entity='swagatk', project=env.spec.id, config={
            'alpha': agent.lr,
            'gamma': agent.gamma,
            'agent': agent.name
        })
    print('Environment name: ', env.spec.id)
    print('RL Agent name:', agent.name)

    if filename is not None:
        file = open(filename, 'w')

    scores = []
    best_score = 0
    for e in range(max_episodes):
        done = False
        score = 0
        state = env.reset()[0]
        while not done:
            action = agent.choose_action(state)
            next_state, reward, done, _, _ = env.step(action)
            agent.store_transitions(state, action, reward)
            score += reward
            state = next_state
            # terminate episode if episodic score is beyond limit
            if ep_max_score is not None and score > ep_max_score: 
                done = True
            if ep_min_score is not None and score < ep_min_score:
                done = True
        # end of while loop
        # train the agent at the end of each episode
        agent.train()
        scores.append(score)
        if score > best_score:
            best_score = score
            agent.save_weights('reinforce_best_model.weights.h5')
            print(f'Best Score: {score}, episode: {e}. Model saved.')
        
        if filename is not None:
            file.write(f'{e}\t{score}\t{np.mean(scores):.2f}\t{np.mean(scores[-100:]):.2f}\n')
            file.flush()
            os.fsync(file.fileno())

        
        if wandb_log:
            wandb.log({
                'episode': e,
                'ep_score': score,
                'mean_score': np.mean(scores),
                'avg100score': np.mean(scores[-100:]),
                'best_score': best_score,
            })

        if e > 100 and np.mean(scores[-100:]) > stop_score:
            print('The problem is solved in {} episodes'.format(e))
            break

        if e % 100== 0:
            print('episode:{}, score: {:.2f}, avgscore: {:.2f}, avg100score: {:.2f}, \
                  Best score: {:.2f}'\
                 .format(e, score, np.mean(scores), \
                         np.mean(scores[-100:]), best_score))
    # end of for-loop
    if filename is not None:
        print('Training completed.')
        file.close()
    if wandb_log:
        run.finish()

######################

# instantiate a gym environment
env = gym.make('CartPole-v1', render_mode='rgb_array')
obs_shape = env.observation_space.shape
action_size = env.action_space.n 

print('Observation shape: ', obs_shape)
print('Action Size: ', action_size)
print('Max Episode steps: ', env.spec.max_episode_steps)

# create an RL agent
agent = REINFORCEAgent(obs_shape, action_size)

# train the RL agent on
train(env, agent, max_episodes=1500, stop_score=499,
       log_freq=100, wandb_log=True)

# validate
# load the best model weights
agent.load_weights('reinforce_best_model.weights.h5')
validate(env, agent, num_episodes=2, gif_file='reinforce_cartpole.gif')


