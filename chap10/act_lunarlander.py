import os
import random
import numpy as np
import tensorflow as tf
from collections import deque
import gymnasium as gym
import wandb

# Force TensorFlow to run smoothly on CPUs/GPUs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def limit_gpu_memory(memory_limit_mb: int) -> None:
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.set_logical_device_configuration(
                    gpu,
                    [tf.config.LogicalDeviceConfiguration(memory_limit=memory_limit_mb)]
                )
            print(f"Limited {len(gpus)} GPU(s) to {memory_limit_mb} MB each.")
        except RuntimeError as e:
            print(f"Error setting GPU memory limit: {e}")

# ==============================================================================
# 1. ARCHITECTURE DEFINITION (ACT Embedded for Self-Contained Execution)
# ==============================================================================

class TransformerDecoderLayer(tf.keras.layers.Layer):
    """
    A standard Transformer Decoder Block matching Chapter 10 specifications.
    Processes target queries with multi-head self-attention and cross-attends to CVAE memory.
    """
    def __init__(self, d_model, num_heads, dff, rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.mha1 = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model)
        self.mha2 = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model)
        self.ffn = tf.keras.Sequential([
            tf.keras.layers.Dense(dff, activation='relu'),
            tf.keras.layers.Dense(d_model)
        ])
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)
        self.dropout3 = tf.keras.layers.Dropout(rate)

    def call(self, x, memory, training=None):
        attn1 = self.mha1(query=x, value=x, key=x, training=training)
        attn1 = self.dropout1(attn1, training=training)
        out1 = self.layernorm1(x + attn1)

        attn2 = self.mha2(query=out1, value=memory, key=memory, training=training)
        attn2 = self.dropout2(attn2, training=training)
        out2 = self.layernorm2(out1 + attn2)

        ffn_output = self.ffn(out2, training=training)
        ffn_output = self.dropout3(ffn_output, training=training)
        out3 = self.layernorm3(out2 + ffn_output)
        return out3


class ACTEncoder(tf.keras.layers.Layer):
    """
    The CVAE Encoder: Encodes state and expert action chunks into a latent space.
    """
    def __init__(self, latent_dim, eff=256, **kwargs):
        super().__init__(**kwargs)
        self.mlp = tf.keras.Sequential([
            tf.keras.layers.Dense(eff, activation='relu'),
            tf.keras.layers.Dense(eff, activation='relu')
        ])
        self.fc_mu = tf.keras.layers.Dense(latent_dim, name="mu")
        self.fc_logvar = tf.keras.layers.Dense(latent_dim, name="logvar")

    def call(self, state, action_chunk):
        batch_size = tf.shape(action_chunk)[0]
        flat_actions = tf.reshape(action_chunk, (batch_size, -1))
        x = tf.concat([state, flat_actions], axis=-1)
        h = self.mlp(x)
        return self.fc_mu(h), self.fc_logvar(h)


class ActionChunkingTransformer(tf.keras.Model):
    """
    The main Action-Chunking Transformer network.
    """
    def __init__(self, state_dim, action_dim, latent_dim=16, chunk_size=20,
                 num_layers=3, d_model=256, num_heads=4, dff=512, eff=256, **kwargs):
        super().__init__(**kwargs)
        self.chunk_size = chunk_size
        self.action_dim = action_dim
        self.latent_dim = latent_dim

        self.encoder = ACTEncoder(latent_dim, eff)
        self.latent_projector = tf.keras.layers.Dense(d_model)
        self.state_projector = tf.keras.layers.Dense(d_model)

        self.query_embed = self.add_weight(
            name="query_embeddings",
            shape=(chunk_size, d_model),
            initializer="glorot_uniform",
            trainable=True
        )

        self.dec_layers = [TransformerDecoderLayer(d_model, num_heads, dff) \
                              for _ in range(num_layers)]
        self.action_head = tf.keras.layers.Dense(action_dim)

    def reparameterize(self, mu, logvar):
        eps = tf.random.normal(shape=tf.shape(mu))
        std = tf.exp(0.5 * logvar)
        return mu + eps * std

    def call(self, inputs, training=None):
        if isinstance(inputs, tuple) or isinstance(inputs, list):
            state, action_chunk = inputs[0], inputs[1]
        else:
            state, action_chunk = inputs, None

        batch_size = tf.shape(state)[0]

        if action_chunk is not None:
            mu, logvar = self.encoder(state, action_chunk)
            z = self.reparameterize(mu, logvar)
            # Analytical KL Loss calculation added explicitly to model losses
            kl_loss = -0.5 * tf.reduce_sum(1.0 + logvar - tf.square(mu) - tf.exp(logvar), axis=-1)
            self.add_loss(tf.reduce_mean(kl_loss))
        else:
            # During evaluation, setting z to the mean of the prior (0) yields the 
            # most likely deterministic action and greatly stabilizes the policy.
            z = tf.zeros(shape=(batch_size, self.latent_dim))

        z_embed = tf.expand_dims(self.latent_projector(z), axis=1)
        state_embed = tf.expand_dims(self.state_projector(state), axis=1)

        memory = tf.concat([state_embed, z_embed], axis=1)

        queries = tf.expand_dims(self.query_embed, axis=0)
        queries = tf.tile(queries, [batch_size, 1, 1])

        out = queries
        for layer in self.dec_layers:
            out = layer(out, memory, training=training)

        # Restrict outputs to [-1, 1] to match LunarLander Continuous action space bounds
        predicted_chunk = tf.tanh(self.action_head(out))
        return predicted_chunk


class TemporalEnsembler:
    """
    Exponentially averages overlapping action sequences to output smooth, continuous motor controls.
    """
    def __init__(self, chunk_size, action_dim, exponential_weight=0.01):
        self.chunk_size = chunk_size
        self.action_dim = action_dim
        self.k = exponential_weight
        self.chunk_queue = deque(maxlen=chunk_size)

    def update_and_get_action(self, new_predicted_chunk):
        self.chunk_queue.append(new_predicted_chunk)
        combined_action = np.zeros(self.action_dim)
        total_weight = 0.0

        for i, chunk in enumerate(reversed(self.chunk_queue)):
            time_idx_within_chunk = i
            weight = np.exp(-self.k * time_idx_within_chunk)
            combined_action += chunk[time_idx_within_chunk] * weight
            total_weight += weight

        return combined_action / total_weight

# ==============================================================================
# 2. EXPERT DATA GENERATION
# ==============================================================================

def pd_expert_policy(s):
    """
    A robust PD controller adapted from the classic OpenAI Gym heuristic.
    This version uses simpler, more stable gains for the continuous action space
    in Gymnasium's LunarLanderContinuous-v3, consistently scoring > 200.
    """
    # State mapping
    x, y, vx, vy, theta, vtheta, left_leg, right_leg = s

    # 1. Calculate Target Angle & Hover Height
    # Target angle is proportional to horizontal position and velocity.
    angle_targ = np.clip(x * 0.5 + vx * 1.0, -0.4, 0.4)
    # Target height is proportional to horizontal distance from center.
    hover_targ = 0.55 * np.abs(x)

    # 2. Calculate PID-like errors/offsets
    angle_todo = (angle_targ - theta) * 0.5 - vtheta * 1.0
    hover_todo = (hover_targ - y) * 0.5 - vy * 0.5

    # 3. Override controls if legs have contact with the ground.
    if left_leg or right_leg:
        angle_todo = 0.0
        hover_todo = -vy * 0.5  # Just need to soften the landing.

    # 4. Map to Continuous Action Space [-1, 1] using the exact Gymnasium scaling
    a = np.array([hover_todo * 20.0 - 1.0, -angle_todo * 20.0], dtype=np.float32)
    return np.clip(a, -1.0, 1.0)


def collect_demonstrations(env, num_episodes=30):
    """
    Simulates the expert policy in the environment and collects high-quality data.
    """
    print(f"Collecting {num_episodes} expert episodes...")
    dataset = []
    expert_rewards = []

    for _ in range(num_episodes):
        obs, _ = env.reset()
        episode_states = []
        episode_actions = []
        done = False
        ep_reward = 0.0

        while not done:
            action = pd_expert_policy(obs)
            episode_states.append(obs)
            episode_actions.append(action)
            obs, reward, terminated, truncated, _ = env.step(action)
            ep_reward += reward
            done = terminated or truncated

        dataset.append({
            'states': np.array(episode_states, dtype=np.float32),
            'actions': np.array(episode_actions, dtype=np.float32)
        })
        expert_rewards.append(ep_reward)
        
    avg_expert_reward = float(np.mean(expert_rewards))
    print(f"Expert Average Reward: {avg_expert_reward:.2f}")
    return dataset, avg_expert_reward

# ==============================================================================
# 3. CHUNKED DATASET GENERATION
# ==============================================================================

def create_training_data(expert_dataset, chunk_size):
    """
    Slices sequential episodic data into discrete states and action chunk targets.
    """
    states_list = []
    chunks_list = []

    for episode in expert_dataset:
        states = episode['states']
        actions = episode['actions']
        num_steps = len(states)

        # We need enough steps to extract a complete action chunk
        if num_steps <= chunk_size:
            continue

        for t in range(num_steps - chunk_size):
            states_list.append(states[t])
            chunks_list.append(actions[t : t + chunk_size])

    return np.array(states_list), np.array(chunks_list)

# ==============================================================================
# 4. TRAINING AND EVALUATION PIPELINE
# ==============================================================================

def evaluate_model(env, model, chunk_size, action_dim, num_episodes=1):
    """Evaluates the model and returns a list of episodic rewards."""
    eval_rewards = []
    for _ in range(num_episodes):
        obs, _ = env.reset()
        ensembler = TemporalEnsembler(chunk_size=chunk_size, action_dim=action_dim, exponential_weight=0.01)
        total_reward = 0.0
        done = False
        while not done:
            state_in = tf.convert_to_tensor(obs[None, :], dtype=tf.float32)
            predicted_chunk = model(state_in, training=False)
            predicted_chunk_np = predicted_chunk[0].numpy()
            smoothed_action = ensembler.update_and_get_action(predicted_chunk_np)
            obs, reward, terminated, truncated, _ = env.step(smoothed_action)
            done = terminated or truncated
            total_reward += reward
        eval_rewards.append(total_reward)
    return eval_rewards

def main():
    # Limit GPU memory to 6 GB
    #limit_gpu_memory(6144)

    # Setup Gymnasium Environment
    env_name = "LunarLanderContinuous-v3"
    env = gym.make(env_name)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    chunk_size = 20 # Slices of 20 steps (planning horizon)

    # 1. Collect demonstration dataset
    expert_data, avg_expert_reward = collect_demonstrations(env, num_episodes=200)

    # 2. Slice expert data into chunks
    train_states, train_chunks = create_training_data(expert_data, chunk_size)
    print(f"Dataset generated. States shape: {train_states.shape}, Chunks shape: {train_chunks.shape}")

    # 3. Instantiate the ACT Agent
    model = ActionChunkingTransformer(state_dim=state_dim, action_dim=action_dim, chunk_size=chunk_size)
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)

    # 4. Train using standard TF Custom Training Loop (combining Reconstruction & KL penalties)
    epochs = 100
    batch_size = 64
    beta = 0.001 # KL Divergence scale factor

    # Initialize Weights & Biases
    wandb.init(
        project="act-lunarlander",
        name="act-training-run",
        config={
            "chunk_size": chunk_size,
            "epochs": epochs,
            "batch_size": batch_size,
            "kl_beta": beta,
            "expert_avg_reward": avg_expert_reward,
        }
    )

    dataset = tf.data.Dataset.from_tensor_slices((train_states, train_chunks)).shuffle(10000).batch(batch_size)

    print("\n--- Starting ACT Training Loop ---")
    historical_rewards = []

    for epoch in range(1, epochs + 1):
        epoch_recon_loss = 0.0
        epoch_kl_loss = 0.0
        steps = 0

        for batch_states, batch_chunks in dataset:
            with tf.GradientTape() as tape:
                # Forward pass: pass (state, target_chunk) to activate training mode
                predictions = model((batch_states, batch_chunks), training=True)

                # Compute L1 Loss for action reconstruction
                recon_loss = tf.reduce_mean(tf.abs(predictions - batch_chunks))

                # Compute KL divergence loss from model context
                kl_loss = tf.reduce_sum(model.losses)

                # Complete ACT Objective: Reconstruction loss + scaled KL penalty
                total_loss = recon_loss + (beta * kl_loss)

            grads = tape.gradient(total_loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

            epoch_recon_loss += recon_loss.numpy()
            epoch_kl_loss += kl_loss.numpy()
            steps += 1

        avg_recon = epoch_recon_loss / steps
        avg_kl = epoch_kl_loss / steps
        training_loss = avg_recon + (beta * avg_kl)
        
        # Evaluate current policy to track episodic rewards
        eval_rewards = evaluate_model(env, model, chunk_size, action_dim, num_episodes=2)
        historical_rewards.extend(eval_rewards)
        
        avg_episodic_reward = np.mean(eval_rewards)
        running_avg_reward = np.mean(historical_rewards[-10:])
        
        print(f"Epoch {epoch:02d}/{epochs} | Recon Loss: {avg_recon:.4f} | KL Loss: {avg_kl:.4f} | Eval Reward: {avg_episodic_reward:.2f}")

        # Log key performance parameters to wandb
        wandb.log({
            "epoch": epoch,
            "training_loss": training_loss,
            "recon_loss": avg_recon,
            "kl_loss": avg_kl,
            "eval_episodic_reward": avg_episodic_reward,
            "avg_eval_episodic_reward": running_avg_reward
        })

    # 5. Live Online Evaluation using Temporal Ensembling
    print("\n--- Evaluating Trained ACT Model with Temporal Ensembling ---")
    eval_episodes = 5
    final_eval_rewards = evaluate_model(env, model, chunk_size, action_dim, num_episodes=eval_episodes)
    
    for ep, total_reward in enumerate(final_eval_rewards):
        print(f"Evaluation Episode {ep+1}: Total Reward = {total_reward:.2f}")

    print(f"\nEvaluation Complete! Average Reward over {eval_episodes} episodes: {np.mean(final_eval_rewards):.2f}")
    env.close()
    wandb.finish()

if __name__ == "__main__":
    main()