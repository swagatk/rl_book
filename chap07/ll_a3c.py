from a3c import run_workers
import multiprocessing
import tensorflow as tf

# create actor & Critic models
def create_actor_model(obs_shape, n_actions):
    s_input = tf.keras.layers.Input(shape=obs_shape)
    x = tf.keras.layers.Dense(128, activation='relu')(s_input)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    a = tf.keras.layers.Dense(n_actions, activation='softmax')(x)
    model = tf.keras.models.Model(s_input, a, name='actor_network')
    model.summary()
    return model

def create_critic_model(obs_shape):
    s_input = tf.keras.layers.Input(shape=obs_shape)
    x = tf.keras.layers.Dense(128, activation='relu')(s_input)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    v = tf.keras.layers.Dense(1, activation=None)(x)
    model = tf.keras.models.Model(s_input, v, name='critic_network')
    model.summary()
    return model

if __name__ == '__main__':
    multiprocessing.set_start_method('spawn')
    run_workers(
        env_name='LunarLander-v3',
        max_num_workers=20,
        max_episodes=1500,
        wandb_log=True,
        max_score=500,
        min_score=-200,
        max_steps=1000,
        create_actor_func=create_actor_model,
        create_critic_func=create_critic_model
    )
