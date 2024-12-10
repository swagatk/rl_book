import gymnasium as gym
from ppo import PPOAgent
from pendu_ppo import ppo_train
import tensorflow as tf
from utils import validate

# create actor & Critic models
def create_actor_model(obs_shape, n_actions):
    s_input = tf.keras.layers.Input(shape=obs_shape)
    x = tf.keras.layers.Dense(512, activation='relu')(s_input)
    x = tf.keras.layers.Dense(512, activation='relu')(x)
    a = tf.keras.layers.Dense(n_actions, activation='tanh')(x)
    model = tf.keras.models.Model(s_input, a, name='actor_network')
    model.summary()
    return model

def create_critic_model(obs_shape, n_actions):
    s_input = tf.keras.layers.Input(shape=obs_shape)
    x = tf.keras.layers.Dense(512, activation='relu')(s_input)
    x = tf.keras.layers.Dense(512, activation='relu')(x)
    v = tf.keras.layers.Dense(1, activation=None)(x)
    model = tf.keras.models.Model(s_input, v, name='critic_network')
    model.summary()
    return model

# wandb config parameters
CFG = dict(
    batch_size=200, 
    entropy_coeff = 0.0,   # required for CLIP method
    c_loss_coeff = 0.0,    # required for CLIP method
    grad_clip = None,
    method = 'clip', # choose between 'clip' or 'penalty'
    kl_target = 0.01, # required for penalty method
    beta = 0.01,  # required for penalty method
    epsilon = 0.3,  # required for clip method
    gamma = 0.99,
    lam = 0.95,     # used for GAE
    buffer_capacity = 20000, # next try with 50000
    lr_a = 1e-3,
    lr_c = 1e-3,
    training_epochs=20,
)

if __name__ == '__main__':
    env = gym.make('LunarLander-v2', continuous=True, render_mode='rgb_array')
    obs_shape = env.observation_space.shape
    action_shape = env.action_space.shape
    action_size = action_shape[0]
    action_ub = env.action_space.high
    action_lb = env.action_space.low
    print('environment name: ', env.spec.name)
    print('Observation shape: ', obs_shape)
    print('Action shape: ', action_shape)
    print('Max episodic Steps: ', env.spec.max_episode_steps)
    print('Action space bounds: ', (action_ub, action_lb))


    # create actor & critic models
    a_model = create_actor_model(obs_shape, action_size)
    c_model = create_critic_model(obs_shape, action_size)


    # create PPO agent
    agent = PPOAgent(obs_shape, action_size, 
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
                    lr_c=CFG['lr_c'],
                    actor_model=a_model,
                    critic_model=c_model)
    
    # train the PPO agent
    ppo_train(env, agent, 
                max_buffer_len=CFG['buffer_capacity'], 
                max_seasons=500,
                epochs=CFG['training_epochs'],
                stop_score=200, max_steps=200, wandb_log=True)

    # validate
    agent.load_weights()
    validate(env, agent, num_episodes=5, gif_file='lunarlander_ppo.gif', max_steps=300)

