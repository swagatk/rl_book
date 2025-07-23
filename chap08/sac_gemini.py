import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_probability as tfp
import gymnasium as gym
from collections import deque
import random
import time

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.get_logger().setLevel('ERROR')

# --- 1. Network Architectures ---

class Actor(keras.Model):
    """
    Actor network for SAC. Outputs mean and log_std for a Gaussian policy.
    The actions are then squashed using tanh to fit the environment's action space.
    """
    def __init__(self, n_actions, name='actor'):
        super(Actor, self).__init__(name=name)
        self.n_actions = n_actions
        self.fc1 = layers.Dense(256, activation='relu')
        self.fc2 = layers.Dense(256, activation='relu')
        # Output mean and log_std for each action
        self.mu = layers.Dense(n_actions, activation=None)
        self.log_std = layers.Dense(n_actions, activation=None)

        # Constraints for log_std to ensure numerical stability and reasonable exploration
        # log_std values are usually clipped to a range like [-20, 2] or [-5, 2]
        self.max_log_std = tf.constant(2.0, dtype=tf.float32)
        self.min_log_std = tf.constant(-20.0, dtype=tf.float32)

    def call(self, state):
        """
        Forward pass for the actor network.
        Args:
            state (tf.Tensor): The current state observation.
        Returns:
            tuple: A tuple containing:
                - mu (tf.Tensor): Mean of the Gaussian distribution.
                - log_std (tf.Tensor): Log standard deviation of the Gaussian distribution.
        """
        x = self.fc1(state)
        x = self.fc2(x)
        mu = self.mu(x)
        log_std = self.log_std(x)
        log_std = tf.clip_by_value(log_std, self.min_log_std, self.max_log_std)
        return mu, log_std

    def sample_action(self, state, reparameterize=True):
        """
        Samples an action from the policy distribution and computes its log probability.
        Args:
            state (tf.Tensor): The current state observation.
            reparameterize (bool): Whether to use the reparameterization trick for sampling.
                                   Set to True for policy updates.
        Returns:
            tuple: A tuple containing:
                - action (tf.Tensor): Sampled action.
                - log_prob (tf.Tensor): Log probability of the sampled action.
        """
        mu, log_std = self.call(state)
        std = tf.exp(log_std)

        # Create a Gaussian distribution
        # tfp.distributions.Normal handles batching automatically
        normal_dist = tfp.distributions.Normal(loc=mu, scale=std)

        if reparameterize:
            # Use reparameterization trick for policy gradient
            z = normal_dist.sample()
        else:
            # For evaluation or interaction with environment, just sample
            z = normal_dist.sample()

        # Apply tanh squashing to map actions to [-1, 1]
        action = tf.tanh(z)

        # Compute log probability of the squashed action
        # This correction term accounts for the change of variables due to tanh
        log_prob = normal_dist.log_prob(z)
        log_prob -= tf.reduce_sum(tf.math.log(1 - action**2 + 1e-6), axis=1, keepdims=True)
        return action, log_prob

class Critic(keras.Model):
    """
    Critic network for SAC. Outputs a Q-value for a given state-action pair.
    SAC uses two critic networks to mitigate overestimation bias.
    """
    def __init__(self, name='critic'):
        super(Critic, self).__init__(name=name)
        self.fc1 = layers.Dense(256, activation='relu')
        self.fc2 = layers.Dense(256, activation='relu')
        self.q = layers.Dense(1, activation=None) # Output a single Q-value

    def call(self, state, action):
        """
        Forward pass for the critic network.
        Args:
            state (tf.Tensor): The current state observation.
            action (tf.Tensor): The action taken.
        Returns:
            tf.Tensor: The estimated Q-value.
        """
        # Concatenate state and action as input to the critic
        x = tf.concat([state, action], axis=-1)
        x = self.fc1(x)
        x = self.fc2(x)
        q_value = self.q(x)
        return q_value

# --- 2. Replay Buffer ---

class ReplayBuffer:
    """
    A simple replay buffer to store experiences (transitions).
    """
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def store(self, state, action, reward, next_state, done):
        """
        Stores a single transition in the buffer.
        Args:
            state (np.array): Current state.
            action (np.array): Action taken.
            reward (float): Reward received.
            next_state (np.array): Next state.
            done (bool): Whether the episode terminated.
        """
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        """
        Samples a random batch of transitions from the buffer.
        Args:
            batch_size (int): Number of transitions to sample.
        Returns:
            tuple: A tuple of numpy arrays for states, actions, rewards, next_states, and dones.
        """
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (np.array(states), np.array(actions), np.array(rewards),
                np.array(next_states), np.array(dones))

    def __len__(self):
        """Returns the current size of the buffer."""
        return len(self.buffer)

# --- 3. SAC Agent ---

class SACAgent:
    """
    Soft Actor-Critic (SAC) Agent implementation.
    """
    def __init__(self, env_name, alpha=0.2, gamma=0.99, tau=0.995,
                 actor_lr=3e-4, critic_lr=3e-4, alpha_lr=3e-4,
                 buffer_capacity=1_000_000, batch_size=256,
                 reward_scale=1.0):
        
        self.env = gym.make(env_name)
        self.state_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.shape[0]
        self.max_action = self.env.action_space.high[0] # Assuming symmetric action space [-max_action, max_action]

        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.reward_scale = reward_scale # Factor to scale rewards, often 1.0 or 5.0 for LunarLander

        # Networks
        self.actor = Actor(self.action_dim)
        self.critic1 = Critic(name='critic1')
        self.critic2 = Critic(name='critic2')
        self.target_critic1 = Critic(name='target_critic1')
        self.target_critic2 = Critic(name='target_critic2')

        # Initialize target networks with the same weights as main critics
        self.target_critic1.set_weights(self.critic1.get_weights())
        self.target_critic2.set_weights(self.critic2.get_weights())

        # Optimizers
        self.actor_optimizer = keras.optimizers.Adam(learning_rate=actor_lr)
        self.critic1_optimizer = keras.optimizers.Adam(learning_rate=critic_lr)
        self.critic2_optimizer = keras.optimizers.Adam(learning_rate=critic_lr)

        # Temperature parameter (alpha) and its optimizer
        self.log_alpha = tf.Variable(tf.math.log(alpha), dtype=tf.float32, trainable=True)
        self.alpha_optimizer = keras.optimizers.Adam(learning_rate=alpha_lr)
        self.target_entropy = -tf.constant(self.action_dim, dtype=tf.float32) # Target entropy for alpha tuning

        # Replay Buffer
        self.replay_buffer = ReplayBuffer(buffer_capacity)

        # Build networks by passing dummy inputs to initialize weights
        self.actor(tf.zeros((1, self.state_dim)))
        self.critic1(tf.zeros((1, self.state_dim)), tf.zeros((1, self.action_dim)))
        self.critic2(tf.zeros((1, self.state_dim)), tf.zeros((1, self.action_dim)))
        self.target_critic1(tf.zeros((1, self.state_dim)), tf.zeros((1, self.action_dim)))
        self.target_critic2(tf.zeros((1, self.state_dim)), tf.zeros((1, self.action_dim)))

    def choose_action(self, observation, evaluate=False):
        """
        Chooses an action based on the current policy.
        Args:
            observation (np.array): Current state observation.
            evaluate (bool): If True, sample deterministically (mean); otherwise, stochastically.
        Returns:
            np.array: The chosen action.
        """
        state = tf.convert_to_tensor([observation], dtype=tf.float32)
        # For evaluation, we typically use the mean of the distribution (deterministic policy)
        # For training, we use reparameterization trick (stochastic policy)
        action, _ = self.actor.sample_action(state, reparameterize=not evaluate)
        return action.numpy()[0] * self.max_action # Scale action to env's range

    @tf.function
    def learn(self):
        """
        Performs one learning step for the SAC agent.
        Updates critic, actor, and optionally the temperature parameter.
        Uses tf.function for graph compilation and performance.
        """
        if len(self.replay_buffer) < self.batch_size:
            return # Not enough samples to learn

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)

        # Convert numpy arrays to TensorFlow tensors
        states = tf.convert_to_tensor(states, dtype=tf.float32)
        actions = tf.convert_to_tensor(actions, dtype=tf.float32)
        rewards = tf.convert_to_tensor(rewards, dtype=tf.float32)
        next_states = tf.convert_to_tensor(next_states, dtype=tf.float32)
        dones = tf.convert_to_tensor(dones, dtype=tf.float32)

        # Reshape rewards and dones for element-wise operations
        rewards = tf.expand_dims(rewards, axis=1)
        dones = tf.expand_dims(dones, axis=1)

        # --- Update Critic Networks ---
        with tf.GradientTape(persistent=True) as tape:
            # Predict next actions and their log probabilities using the current actor
            next_actions, next_log_probs = self.actor.sample_action(next_states)

            # Compute target Q-values using target critics
            q1_target_next = self.target_critic1(next_states, next_actions)
            q2_target_next = self.target_critic2(next_states, next_actions)
            min_q_target_next = tf.minimum(q1_target_next, q2_target_next)

            # Soft Bellman backup equation for target Q-value
            # y = r + gamma * (1 - done) * (min_Q_target(s', a') - alpha * log_pi(a'|s'))
            target_q_values = self.reward_scale * rewards + self.gamma * (1 - dones) * \
                              (min_q_target_next - tf.exp(self.log_alpha) * next_log_probs)

            # Predict current Q-values
            q1 = self.critic1(states, actions)
            q2 = self.critic2(states, actions)

            # Critic loss (MSE between predicted Q and target Q)
            critic1_loss = tf.reduce_mean(tf.square(q1 - target_q_values))
            critic2_loss = tf.reduce_mean(tf.square(q2 - target_q_values))

        # Apply gradients for critics
        critic1_grads = tape.gradient(critic1_loss, self.critic1.trainable_variables)
        self.critic1_optimizer.apply_gradients(zip(critic1_grads, self.critic1.trainable_variables))
        critic2_grads = tape.gradient(critic2_loss, self.critic2.trainable_variables)
        self.critic2_optimizer.apply_gradients(zip(critic2_grads, self.critic2.trainable_variables))

        # --- Update Actor Network and Alpha ---
        with tf.GradientTape(persistent=True) as tape:
            # Sample actions and log probabilities from the current policy for actor update
            current_actions, current_log_probs = self.actor.sample_action(states)

            # Compute Q-values from the main critics for the current actions
            q1_current = self.critic1(states, current_actions)
            q2_current = self.critic2(states, current_actions)
            min_q_current = tf.minimum(q1_current, q2_current)

            # Actor loss (maximize soft Q-value, incorporating entropy)
            # J_pi = E_s,a~pi [alpha * log_pi(a|s) - Q(s,a)] -> minimize -J_pi
            actor_loss = tf.reduce_mean(tf.exp(self.log_alpha) * current_log_probs - min_q_current)

            # Alpha loss (minimize difference between current entropy and target entropy)
            # J_alpha = E_s,a~pi [-alpha * (log_pi(a|s) + H_0)]
            alpha_loss = tf.reduce_mean(-self.log_alpha * (current_log_probs + self.target_entropy))

        # Apply gradients for actor
        actor_grads = tape.gradient(actor_loss, self.actor.trainable_variables)
        self.actor_optimizer.apply_gradients(zip(actor_grads, self.actor.trainable_variables))

        # Apply gradients for alpha
        alpha_grads = tape.gradient(alpha_loss, [self.log_alpha])
        self.alpha_optimizer.apply_gradients(zip(alpha_grads, [self.log_alpha]))

        # --- Update Target Networks (Polyak Averaging) ---
        for target_param, param in zip(self.target_critic1.trainable_variables, self.critic1.trainable_variables):
            target_param.assign(target_param * self.tau + param * (1 - self.tau))
        for target_param, param in zip(self.target_critic2.trainable_variables, self.critic2.trainable_variables):
            target_param.assign(target_param * self.tau + param * (1 - self.tau))

# --- 4. Training Loop ---

def train_sac(env_name='LunarLanderContinuous-v3', num_episodes=1000,
              warmup_steps=1000, update_after_steps=1000, updates_per_step=1):
    """
    Main training function for the SAC agent.
    Args:
        env_name (str): Name of the Gymnasium environment.
        num_episodes (int): Number of episodes to train for.
        warmup_steps (int): Number of initial random steps to fill replay buffer.
        update_after_steps (int): Start learning after this many steps.
        updates_per_step (int): Number of gradient updates per environment step.
    """
    agent = SACAgent(env_name)
    total_steps = 0
    scores = []
    best_score = -np.inf

    print(f"Training SAC on {env_name} for {num_episodes} episodes...")
    print(f"Warmup steps: {warmup_steps}")
    print(f"Learning starts after: {update_after_steps} steps")

    for episode in range(1, num_episodes + 1):
        state, info = agent.env.reset()
        done = False
        truncated = False
        episode_score = 0
        episode_steps = 0

        while not done and not truncated:
            # Perform initial random actions for warmup
            if total_steps < warmup_steps:
                action = agent.env.action_space.sample()
            else:
                action = agent.choose_action(state)

            next_state, reward, done, truncated, info = agent.env.step(action)
            agent.replay_buffer.store(state, action / agent.max_action, reward, next_state, done) # Store normalized action

            state = next_state
            episode_score += reward
            total_steps += 1
            episode_steps += 1

            # Start learning after warmup and minimum buffer size
            if total_steps >= update_after_steps:
                for _ in range(updates_per_step):
                    agent.learn()

        scores.append(episode_score)
        avg_score = np.mean(scores[-100:]) # Average over last 100 episodes for stability

        if avg_score > best_score:
            best_score = avg_score
            # Optional: Save model weights here if desired
            # agent.actor.save_weights('sac_actor_best.h5')
            # agent.critic1.save_weights('sac_critic1_best.h5')
            # agent.critic2.save_weights('sac_critic2_best.h5')

        print(f"Episode {episode:4d} | Total Steps: {total_steps:7d} | Episode Steps: {episode_steps:4d} | "
              f"Score: {episode_score:8.2f} | Avg 100-ep Score: {avg_score:8.2f} | Alpha: {tf.exp(agent.log_alpha).numpy():.4f}")

        # LunarLander-v3 is considered solved if average reward over 100 episodes is >= 200
        if avg_score >= 200 and episode >= 100:
            print(f"\nEnvironment solved in {episode} episodes!")
            break

    print("\nTraining finished.")
    # Optional: Evaluate the final policy
    # evaluate_policy(agent, agent.env, num_eval_episodes=10)

def evaluate_policy(agent, env, num_eval_episodes=5):
    """
    Evaluates the trained SAC policy.
    Args:
        agent (SACAgent): The trained SAC agent.
        env (gym.Env): The environment to evaluate in.
        num_eval_episodes (int): Number of episodes for evaluation.
    """
    eval_scores = []
    print(f"\nEvaluating policy for {num_eval_episodes} episodes...")
    for i in range(num_eval_episodes):
        state, info = env.reset()
        done = False
        truncated = False
        episode_score = 0
        while not done and not truncated:
            action = agent.choose_action(state, evaluate=True) # Use deterministic action for evaluation
            state, reward, done, truncated, info = env.step(action)
            episode_score += reward
            # Optional: env.render() if you want to visualize
        eval_scores.append(episode_score)
        print(f"  Eval Episode {i+1}: Score = {episode_score:.2f}")
    print(f"Average evaluation score: {np.mean(eval_scores):.2f}")


if __name__ == '__main__':
    # Set random seeds for reproducibility
    tf.random.set_seed(42)
    np.random.seed(42)
    random.seed(42)

    # Run the training
    train_sac(num_episodes=2000, warmup_steps=10000, update_after_steps=10000, updates_per_step=1)

    # After training, you can load the best weights and evaluate
    # agent = SACAgent('LunarLander-v3')
    # agent.actor.load_weights('sac_actor_best.h5')
    # agent.critic1.load_weights('sac_critic1_best.h5')
    # agent.critic2.load_weights('sac_critic2_best.h5')
    # evaluate_policy(agent, agent.env, num_eval_episodes=10)
