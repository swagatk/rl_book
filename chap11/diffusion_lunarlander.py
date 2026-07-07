import os
import random
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import tensorflow as tf
import wandb


# Keep TensorFlow logs clean.
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


@dataclass
class Config:
	env_name: str = "LunarLanderContinuous-v3"
	seed: int = 42

	# Data settings
	expert_episodes: int = 200
	val_split: float = 0.1

	# Diffusion settings
	diffusion_steps: int = 32
	beta_start: float = 1e-4
	beta_end: float = 2e-2

	# Model/optimization settings
	hidden_dim: int = 256
	time_embed_dim: int = 64
	learning_rate: float = 1e-4
	epochs: int = 200
	batch_size: int = 256

	# Evaluation settings
	eval_every: int = 5
	eval_episodes: int = 3
	final_eval_episodes: int = 5
	final_gif_name: str = "diffusion_lunarlander_final.gif"
	final_gif_fps: int = 30

	# W&B
	wandb_project: str = "diffusion-lunarlander"
	wandb_run_name: str = "diffusion-policy-run"


def set_seed(seed: int) -> None:
	random.seed(seed)
	np.random.seed(seed)
	tf.random.set_seed(seed)


def pd_expert_policy(state: np.ndarray) -> np.ndarray:
	"""
	Simple PD-style expert adapted for LunarLanderContinuous-v3.
	Returns 2D continuous action in [-1, 1].
	"""
	x, y, vx, vy, theta, vtheta, left_leg, right_leg = state

	angle_target = np.clip(0.5 * x + 1.0 * vx, -0.4, 0.4)
	hover_target = 0.55 * np.abs(x)

	angle_ctrl = (angle_target - theta) * 0.5 - 1.0 * vtheta
	hover_ctrl = (hover_target - y) * 0.5 - 0.5 * vy

	if left_leg or right_leg:
		angle_ctrl = 0.0
		hover_ctrl = -0.5 * vy

	action = np.array([hover_ctrl * 20.0 - 1.0, -angle_ctrl * 20.0], dtype=np.float32)
	return np.clip(action, -1.0, 1.0)


def collect_expert_dataset(env: gym.Env, episodes: int) -> tuple[np.ndarray, np.ndarray, float]:
	"""
	Collects state-action pairs from the expert policy.
	"""
	all_states = []
	all_actions = []
	episode_rewards = []

	print(f"Collecting expert data from {episodes} episodes...")
	for ep in range(episodes):
		state, _ = env.reset()
		done = False
		total_reward = 0.0

		while not done:
			action = pd_expert_policy(state)
			all_states.append(state)
			all_actions.append(action)

			state, reward, terminated, truncated, _ = env.step(action)
			done = terminated or truncated
			total_reward += float(reward)

		episode_rewards.append(total_reward)
		if (ep + 1) % 20 == 0:
			print(f"  episode {ep + 1:03d}/{episodes} | reward: {total_reward:.2f}")

	avg_reward = float(np.mean(episode_rewards))
	print(f"Expert avg reward: {avg_reward:.2f}")

	states = np.asarray(all_states, dtype=np.float32)
	actions = np.asarray(all_actions, dtype=np.float32)
	return states, actions, avg_reward


def train_val_split(
	states: np.ndarray,
	actions: np.ndarray,
	val_split: float,
	seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
	num_samples = states.shape[0]
	idx = np.arange(num_samples)
	rng = np.random.default_rng(seed)
	rng.shuffle(idx)

	val_size = int(num_samples * val_split)
	val_idx = idx[:val_size]
	train_idx = idx[val_size:]

	train_states = states[train_idx]
	train_actions = actions[train_idx]
	val_states = states[val_idx]
	val_actions = actions[val_idx]
	return train_states, train_actions, val_states, val_actions


def sinusoidal_time_embedding(timesteps: tf.Tensor, dim: int) -> tf.Tensor:
	"""
	Standard sinusoidal embedding for diffusion timestep conditioning.
	"""
	timesteps = tf.cast(timesteps, tf.float32)
	half = dim // 2
	freq = tf.exp(
		-tf.math.log(10000.0) * tf.cast(tf.range(half), tf.float32) / tf.cast(max(half - 1, 1), tf.float32)
	)
	angles = tf.expand_dims(timesteps, axis=1) * tf.expand_dims(freq, axis=0)
	emb = tf.concat([tf.sin(angles), tf.cos(angles)], axis=1)
	if dim % 2 == 1:
		emb = tf.pad(emb, [[0, 0], [0, 1]])
	return emb


class DiffusionPolicy(tf.keras.Model):
	"""
	Small MLP denoiser that predicts Gaussian noise epsilon from:
	- current state
	- noised action sample
	- diffusion timestep
	"""

	def __init__(self, state_dim: int, action_dim: int, hidden_dim: int, time_embed_dim: int):
		super().__init__()
		self.time_embed_dim = time_embed_dim

		self.net = tf.keras.Sequential(
			[
				tf.keras.layers.Dense(hidden_dim, activation="relu"),
				tf.keras.layers.Dense(hidden_dim, activation="relu"),
				tf.keras.layers.Dense(hidden_dim, activation="relu"),
				tf.keras.layers.Dense(action_dim),
			]
		)

		# Build early so summary/weights are created with clear expected shapes.
		dummy_state = tf.zeros((1, state_dim), dtype=tf.float32)
		dummy_action = tf.zeros((1, action_dim), dtype=tf.float32)
		dummy_t = tf.zeros((1,), dtype=tf.int32)
		_ = self((dummy_state, dummy_action, dummy_t), training=False)

	def call(self, inputs, training=None):
		states, noisy_actions, timesteps = inputs
		t_emb = sinusoidal_time_embedding(timesteps, self.time_embed_dim)
		x = tf.concat([states, noisy_actions, t_emb], axis=-1)
		return self.net(x, training=training)


class DiffusionSchedule:
	"""
	DDPM forward/backward schedule with linear beta noise schedule.
	"""

	def __init__(self, num_steps: int, beta_start: float, beta_end: float):
		self.num_steps = num_steps

		betas = np.linspace(beta_start, beta_end, num_steps, dtype=np.float32)
		alphas = 1.0 - betas
		alpha_bars = np.cumprod(alphas, axis=0)

		self.betas = tf.constant(betas, dtype=tf.float32)
		self.alphas = tf.constant(alphas, dtype=tf.float32)
		self.alpha_bars = tf.constant(alpha_bars, dtype=tf.float32)

		self.sqrt_alpha_bars = tf.constant(np.sqrt(alpha_bars), dtype=tf.float32)
		self.sqrt_one_minus_alpha_bars = tf.constant(np.sqrt(1.0 - alpha_bars), dtype=tf.float32)


def _extract_by_t(coeffs: tf.Tensor, timesteps: tf.Tensor) -> tf.Tensor:
	"""
	Gathers per-sample scalar coefficients and reshapes for broadcast over action dims.
	"""
	values = tf.gather(coeffs, timesteps)
	return tf.expand_dims(values, axis=1)


def q_sample(
	clean_actions: tf.Tensor,
	timesteps: tf.Tensor,
	noise: tf.Tensor,
	schedule: DiffusionSchedule,
) -> tf.Tensor:
	sqrt_ab = _extract_by_t(schedule.sqrt_alpha_bars, timesteps)
	sqrt_omb = _extract_by_t(schedule.sqrt_one_minus_alpha_bars, timesteps)
	return sqrt_ab * clean_actions + sqrt_omb * noise


@tf.function
def train_step(
	model: tf.keras.Model,
	optimizer: tf.keras.optimizers.Optimizer,
	schedule: DiffusionSchedule,
	states: tf.Tensor,
	actions: tf.Tensor,
) -> tf.Tensor:
	batch_size = tf.shape(states)[0]
	t = tf.random.uniform(tf.expand_dims(batch_size, axis=0), minval=0, maxval=schedule.num_steps, dtype=tf.int32)
	noise = tf.random.normal(tf.shape(actions))
	noisy_actions = q_sample(actions, t, noise, schedule)

	with tf.GradientTape() as tape:
		pred_noise = model((states, noisy_actions, t), training=True)
		loss = tf.reduce_mean(tf.square(noise - pred_noise))

	grads = tape.gradient(loss, model.trainable_variables)
	optimizer.apply_gradients(zip(grads, model.trainable_variables))
	return loss


@tf.function
def val_step(
	model: tf.keras.Model,
	schedule: DiffusionSchedule,
	states: tf.Tensor,
	actions: tf.Tensor,
) -> tf.Tensor:
	batch_size = tf.shape(states)[0]
	t = tf.random.uniform(tf.expand_dims(batch_size, axis=0), minval=0, maxval=schedule.num_steps, dtype=tf.int32)
	noise = tf.random.normal(tf.shape(actions))
	noisy_actions = q_sample(actions, t, noise, schedule)

	pred_noise = model((states, noisy_actions, t), training=False)
	loss = tf.reduce_mean(tf.square(noise - pred_noise))
	return loss


def sample_action(
	model: tf.keras.Model,
	schedule: DiffusionSchedule,
	state: np.ndarray,
	action_dim: int,
) -> np.ndarray:
	"""
	Reverse diffusion: start from Gaussian noise and denoise for T steps.
	"""
	x = tf.random.normal((1, action_dim), dtype=tf.float32)
	state_t = tf.convert_to_tensor(state[None, :], dtype=tf.float32)

	for t in reversed(range(schedule.num_steps)):
		t_batch = tf.constant([t], dtype=tf.int32)
		eps = model((state_t, x, t_batch), training=False)

		alpha_t = schedule.alphas[t]
		alpha_bar_t = schedule.alpha_bars[t]
		beta_t = schedule.betas[t]

		# DDPM mean update.
		mean = (1.0 / tf.sqrt(alpha_t)) * (x - ((1.0 - alpha_t) / tf.sqrt(1.0 - alpha_bar_t)) * eps)

		if t > 0:
			z = tf.random.normal(tf.shape(x), dtype=tf.float32)
			x = mean + tf.sqrt(beta_t) * z
		else:
			x = mean

	return tf.clip_by_value(x[0], -1.0, 1.0).numpy()


def evaluate_policy(
	env: gym.Env,
	model: tf.keras.Model,
	schedule: DiffusionSchedule,
	action_dim: int,
	num_episodes: int,
) -> tuple[float, list[float]]:
	rewards = []
	for _ in range(num_episodes):
		state, _ = env.reset()
		done = False
		total_reward = 0.0

		while not done:
			action = sample_action(model, schedule, state, action_dim)
			state, reward, terminated, truncated, _ = env.step(action)
			done = terminated or truncated
			total_reward += float(reward)

		rewards.append(total_reward)

	return float(np.mean(rewards)), rewards


def evaluate_policy_and_save_gif(
	env: gym.Env,
	model: tf.keras.Model,
	schedule: DiffusionSchedule,
	action_dim: int,
	num_episodes: int,
	gif_path: str,
	fps: int = 30,
) -> tuple[float, list[float], bool]:
	"""
	Evaluates the policy and records the first episode as a GIF.
	Requires env to support rgb-array rendering.
	"""
	try:
		import imageio.v2 as imageio
	except ImportError:
		print("imageio is not installed; skipping GIF export.")
		avg_reward, rewards = evaluate_policy(env, model, schedule, action_dim, num_episodes)
		return avg_reward, rewards, False

	rewards = []
	frames = []

	for ep in range(num_episodes):
		state, _ = env.reset()
		done = False
		total_reward = 0.0

		if ep == 0:
			frame = env.render()
			if frame is not None:
				frames.append(frame)

		while not done:
			action = sample_action(model, schedule, state, action_dim)
			state, reward, terminated, truncated, _ = env.step(action)
			done = terminated or truncated
			total_reward += float(reward)

			if ep == 0:
				frame = env.render()
				if frame is not None:
					frames.append(frame)

		rewards.append(total_reward)

	os.makedirs(os.path.dirname(gif_path), exist_ok=True)
	if frames:
		imageio.mimsave(gif_path, frames, fps=fps)
		return float(np.mean(rewards)), rewards, True

	print("No frames captured; skipping GIF export.")
	return float(np.mean(rewards)), rewards, False


def make_dataset(states: np.ndarray, actions: np.ndarray, batch_size: int, shuffle: bool) -> tf.data.Dataset:
	ds = tf.data.Dataset.from_tensor_slices((states, actions))
	if shuffle:
		ds = ds.shuffle(min(len(states), 100000), reshuffle_each_iteration=True)
	ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
	return ds


def limit_gpu_memory(memory_limit_mb: int = 9216) -> None:
    gpus = tf.config.list_physical_devices("GPU")
    if not gpus:
        print("No GPU found.")
        return
    try:
        for gpu in gpus:
            tf.config.set_logical_device_configuration(
                gpu,
                [tf.config.LogicalDeviceConfiguration(memory_limit=memory_limit_mb)],
            )
        print(f"Limited {len(gpus)} GPU(s) to {memory_limit_mb} MB each.")
    except RuntimeError as e:
        # Happens if TF already initialized the GPU.
        print(f"Could not set GPU memory limit: {e}")


def main() -> None:
	cfg = Config()
	set_seed(cfg.seed)
	limit_gpu_memory(9216)
	env = gym.make(cfg.env_name)
	obs_shape = env.observation_space.shape
	act_shape = env.action_space.shape
	if obs_shape is None or act_shape is None:
		raise ValueError("Environment observation/action shape cannot be None.")
	state_dim = int(obs_shape[0])
	action_dim = int(act_shape[0])

	# 1) Gather expert transitions.
	states, actions, expert_avg_reward = collect_expert_dataset(env, episodes=cfg.expert_episodes)

	# 2) Train/validation split.
	train_s, train_a, val_s, val_a = train_val_split(
		states,
		actions,
		val_split=cfg.val_split,
		seed=cfg.seed,
	)
	print(
		f"Dataset sizes | train: {len(train_s)} transitions | "
		f"val: {len(val_s)} transitions"
	)

	train_ds = make_dataset(train_s, train_a, cfg.batch_size, shuffle=True)
	val_ds = make_dataset(val_s, val_a, cfg.batch_size, shuffle=False)

	# 3) Build diffusion policy and optimizer.
	model = DiffusionPolicy(
		state_dim=state_dim,
		action_dim=action_dim,
		hidden_dim=cfg.hidden_dim,
		time_embed_dim=cfg.time_embed_dim,
	)
	schedule = DiffusionSchedule(
		num_steps=cfg.diffusion_steps,
		beta_start=cfg.beta_start,
		beta_end=cfg.beta_end,
	)
	optimizer = tf.keras.optimizers.Adam(learning_rate=cfg.learning_rate)

	wandb.init(
		project=cfg.wandb_project,
		name=cfg.wandb_run_name,
		config={
			"env": cfg.env_name,
			"seed": cfg.seed,
			"expert_episodes": cfg.expert_episodes,
			"train_transitions": int(len(train_s)),
			"val_transitions": int(len(val_s)),
			"diffusion_steps": cfg.diffusion_steps,
			"beta_start": cfg.beta_start,
			"beta_end": cfg.beta_end,
			"hidden_dim": cfg.hidden_dim,
			"time_embed_dim": cfg.time_embed_dim,
			"learning_rate": cfg.learning_rate,
			"epochs": cfg.epochs,
			"batch_size": cfg.batch_size,
			"eval_every": cfg.eval_every,
			"eval_episodes": cfg.eval_episodes,
			"expert_avg_reward": expert_avg_reward,
		},
	)

	print("\n--- Starting Diffusion Policy Training ---")

	running_eval_rewards = []
	for epoch in range(1, cfg.epochs + 1):
		train_losses = []
		for batch_states, batch_actions in train_ds:
			loss = train_step(model, optimizer, schedule, batch_states, batch_actions)
			train_losses.append(float(loss.numpy()))
		train_loss = float(np.mean(train_losses))

		val_losses = []
		for batch_states, batch_actions in val_ds:
			loss = val_step(model, schedule, batch_states, batch_actions)
			val_losses.append(float(loss.numpy()))
		val_loss = float(np.mean(val_losses)) if val_losses else float("nan")

		metrics = {
			"epoch": epoch,
			"train_denoise_mse": train_loss,
			"val_denoise_mse": val_loss,
		}

		# Evaluate in environment periodically to track policy quality.
		if epoch % cfg.eval_every == 0 or epoch == 1 or epoch == cfg.epochs:
			eval_avg_reward, eval_rewards = evaluate_policy(
				env=env,
				model=model,
				schedule=schedule,
				action_dim=action_dim,
				num_episodes=cfg.eval_episodes,
			)
			running_eval_rewards.extend(eval_rewards)
			metrics["eval_avg_reward"] = eval_avg_reward
			metrics["eval_running_avg_reward_10"] = float(np.mean(running_eval_rewards[-10:]))

			print(
				f"Epoch {epoch:03d}/{cfg.epochs} | "
				f"train_mse: {train_loss:.6f} | val_mse: {val_loss:.6f} | "
				f"eval_reward: {eval_avg_reward:.2f}"
			)
		else:
			print(
				f"Epoch {epoch:03d}/{cfg.epochs} | "
				f"train_mse: {train_loss:.6f} | val_mse: {val_loss:.6f}"
			)

		wandb.log(metrics)

	print("\n--- Final Evaluation ---")
	final_gif_path = os.path.join(os.path.dirname(__file__), cfg.final_gif_name)
	final_eval_env = gym.make(cfg.env_name, render_mode="rgb_array")
	final_avg_reward, final_rewards, gif_saved = evaluate_policy_and_save_gif(
		env=final_eval_env,
		model=model,
		schedule=schedule,
		action_dim=action_dim,
		num_episodes=cfg.final_eval_episodes,
		gif_path=final_gif_path,
		fps=cfg.final_gif_fps,
	)
	final_eval_env.close()
	for i, r in enumerate(final_rewards, start=1):
		print(f"Episode {i}: reward = {r:.2f}")
	print(f"Final average reward over {cfg.final_eval_episodes} episodes: {final_avg_reward:.2f}")
	if gif_saved:
		print(f"Saved final evaluation GIF to: {final_gif_path}")
	else:
		print("Final evaluation GIF was not saved.")

	wandb.log({"final_eval_avg_reward": final_avg_reward})
	if gif_saved:
		wandb.log({"final_eval_rollout_gif": wandb.Video(final_gif_path, format="gif")})
	wandb.finish()
	env.close()


if __name__ == "__main__":
	main()
