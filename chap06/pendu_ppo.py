import gymnasium as gym
from ppo import PPOAgent
import numpy as np
import wandb

# function to collect trajectories
def collect_trajectories(env, agent, tmax=1000, max_steps=200):
    states, next_states, actions = [], [], []
    rewards, dones = [], []
    ep_count = 0        # episode count
    state = env.reset()[0]
    step = 0
    for t in range(tmax):
        step += 1
        action = agent.policy(state)
        next_state, reward, done, _, _ = env.step(action)
        states.append(state)
        actions.append(action)
        next_states.append(next_state)
        rewards.append(reward)
        dones.append(done)
        state = next_state
        
        if max_steps is not None and step > max_steps:
            done = True
        
        if done:
            ep_count += 1
            state = env.reset()[0]
            step = 0
    return states, actions, rewards, next_states, dones, ep_count

## Train function
def ppo_train(env, agent, max_buffer_len=1000, max_seasons=100, epochs=20,
             max_steps=None, stop_score=None, wandb_log=True):
    print('Environment name: ', env.spec.name)
    print('RL Agent name:', agent.name)
    if wandb_log:
        run = wandb.init(entity='swagatk', project=env.spec.name, config=CFG)
        
    #training
    best_score = -np.inf
    season_scores = []
    total_ep_cnt = 0
    for s in range(max_seasons):
        # collect trajectories
        states, actions, rewards, next_states, dones, ep_count = \
            collect_trajectories(env, agent, tmax=max_buffer_len, max_steps=max_steps)
        
        total_ep_cnt += (ep_count+1)

        # train the agent
        a_loss, c_loss, kld_value = agent.train(states, actions, rewards,
                                                next_states, dones, epochs=epochs)

        season_score = np.sum(rewards, axis=0) / (ep_count + 1)
        season_scores.append(season_score)
        
        if season_score > best_score:
            best_score = season_score
            agent.save_weights()
        
        # logging
        if wandb_log:
            wandb.log({
                        'season_score': season_score,
                        'actor_loss': a_loss,
                        'critic_loss': c_loss,
                        'kld_value' : kld_value, 
                        'penalty_coeff':agent.actor.beta,
                        'ep_count' : total_ep_cnt,
                    })
            
        if stop_score is not None and season_score > stop_score:
            print(f'Problem is solved in {s} seasons or {total_ep_cnt} episodes.' )
            break
        print(f'season: {s}, episodes: {total_ep_cnt}, season_score: {season_score:.2f}, avg_ep_reward: {np.mean(season_scores):.2f}, best_score: {best_score:.2f}')
    
    if wandb_log:
        run.finish()


if __name__ == '__main__':
    env = gym.make('Pendulum-v1')
    obs_shape = env.observation_space.shape
    action_shape = env.action_space.shape
    action_size = action_shape[0]
    action_ub = env.action_space.high
    action_lb = env.action_space.low
    print('Observation shape: ', obs_shape)
    print('Action shape: ', action_shape)
    print('Max episodic Steps: ', env.spec.max_episode_steps)
    print('Action space bounds: ', action_ub, action_lb)

    # wandb configuration
    CFG = dict(
        batch_size=200,
        entropy_coeff = 0.01,   # required for CLIP method
        c_loss_coeff = 0.01,    # required for CLIP method
        grad_clip = None,
        method = 'penalty', # choose between 'clip' or 'penalty'
        kl_target = 0.01,    # 0.01
        beta = 0.01,  # required for penalty method
        epsilon = 0.1,  # required for clip method
        gamma = 0.99,
        lam = 0.95,     # used for GAE
        buffer_capacity = 10000, # 10000 works
        lr_a = 1e-4,
        lr_c = 2e-4,
        training_epochs=20,
    )

    # create a ppo agent
    ppo_agent = PPOAgent(obs_shape, action_size, 
                     batch_size=CFG['batch_size'], 
                     action_upper_bound=action_ub,
                     entropy_coeff=CFG['entropy_coeff'],
                     c_loss_coeff=CFG['c_loss_coeff'],
                     kl_target=CFG['kl_target'],
                     gamma=CFG['gamma'],
                     beta=CFG['beta'],
                     epsilon=CFG['epsilon'],
                     lmbda=CFG['lam'],
                    grad_clip=CFG['grad_clip'],
                    method=CFG['method'],
                    lr_a=CFG['lr_a'],
                    lr_c=CFG['lr_c'])
    
    # train the agent
    ppo_train(env, ppo_agent, max_buffer_len=CFG['buffer_capacity'], max_seasons=100, epochs=20,
             max_steps=200, stop_score=-200, wandb_log=True)
    


