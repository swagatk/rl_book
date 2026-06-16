"""
Action Chunking Transformer (ACT) for Panda cube pick-and-place in Gymnasium.

This script trains an ACT policy from scripted expert demonstrations and then
evaluates it with chunked action rollout.

Example:
	python chap10/act.py --train-episodes 250 --epochs 80 --eval-episodes 20

Dependencies:
	pip install tensorflow gymnasium panda-gym numpy
"""

import argparse
import collections
import os
import random
from dataclasses import dataclass
from typing import Any, Callable, Deque, Dict, List, Sequence, Tuple

import gymnasium as gym
import numpy as np
import tensorflow as tf

try:
	import wandb as _wandb
	wandb: Any = _wandb
except ImportError:
	wandb = None


def wandb_is_active() -> bool:
	return wandb is not None and getattr(wandb, "run", None) is not None


def set_seed(seed: int) -> None:
	random.seed(seed)
	np.random.seed(seed)
	tf.random.set_seed(seed)
	os.environ["PYTHONHASHSEED"] = str(seed)

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


def register_panda_envs() -> None:
	# panda-gym registers envs on import.
	import panda_gym  # noqa: F401


def extract_obs_vector(obs: Dict[str, np.ndarray]) -> np.ndarray:
	parts = [
		np.asarray(obs["observation"], dtype=np.float32),
		np.asarray(obs["achieved_goal"], dtype=np.float32),
		np.asarray(obs["desired_goal"], dtype=np.float32),
	]
	return np.concatenate(parts, axis=0)


def try_extract_gripper_width(obs: Dict[str, np.ndarray]) -> float:
	base = np.asarray(obs["observation"], dtype=np.float32)
	if base.shape[0] > 6:
		return float(base[6])
	return 0.04


def scripted_pick_expert(
	obs: Dict[str, np.ndarray],
	action_low: np.ndarray,
	action_high: np.ndarray,
) -> np.ndarray:
	vec = np.asarray(obs["observation"], dtype=np.float32)
	gripper_pos = vec[:3]
	object_pos = np.asarray(obs["achieved_goal"], dtype=np.float32)
	goal_pos = np.asarray(obs["desired_goal"], dtype=np.float32)
	gripper_width = try_extract_gripper_width(obs)

	above_object = object_pos + np.array([0.0, 0.0, 0.08], dtype=np.float32)
	lift_target = object_pos + np.array([0.0, 0.0, 0.15], dtype=np.float32)
	above_goal = goal_pos + np.array([0.0, 0.0, 0.10], dtype=np.float32)

	d_to_above_obj = np.linalg.norm(gripper_pos - above_object)
	d_to_obj = np.linalg.norm(gripper_pos - object_pos)
	d_obj_to_goal = np.linalg.norm(object_pos - goal_pos)

	carrying = d_to_obj < 0.05 and gripper_width < 0.05

	if not carrying and d_to_above_obj > 0.04:
		pos_target = above_object
		grip = 1.0
	elif not carrying and d_to_obj > 0.015:
		pos_target = object_pos
		grip = 1.0
	elif not carrying:
		pos_target = object_pos
		grip = -1.0
	elif d_obj_to_goal > 0.06:
		if gripper_pos[2] < lift_target[2] - 0.02:
			pos_target = lift_target
		elif np.linalg.norm(gripper_pos - above_goal) > 0.04:
			pos_target = above_goal
		else:
			pos_target = goal_pos
		grip = -1.0
	else:
		pos_target = goal_pos
		grip = 1.0

	delta = 6.0 * (pos_target - gripper_pos)
	action = np.zeros_like(action_high, dtype=np.float32)
	action[:3] = delta
	if action.shape[0] > 3:
		action[3] = grip

	action = np.clip(action, action_low, action_high)
	return action


@dataclass
class EpisodeRecord:
	observations: List[np.ndarray]
	actions: List[np.ndarray]


def collect_expert_dataset(
	env: gym.Env,
	num_episodes: int,
	max_steps: int,
	action_noise: float,
) -> Tuple[List[EpisodeRecord], int]:
	records: List[EpisodeRecord] = []
	success_count = 0

	a_low = np.asarray(env.action_space.low, dtype=np.float32)
	a_high = np.asarray(env.action_space.high, dtype=np.float32)

	for _ in range(num_episodes):
		obs, _ = env.reset()
		observations: List[np.ndarray] = []
		actions: List[np.ndarray] = []
		success = False

		for _step in range(max_steps):
			obs_vec = extract_obs_vector(obs)
			action = scripted_pick_expert(obs, a_low, a_high)
			if action_noise > 0.0:
				action = action + np.random.normal(0.0, action_noise, size=action.shape)
				action = np.clip(action, a_low, a_high)

			next_obs, reward, terminated, truncated, info = env.step(action)
			_ = reward

			observations.append(obs_vec)
			actions.append(action.astype(np.float32))

			success = success or bool(info.get("is_success", False))
			obs = next_obs
			if terminated or truncated:
				break

		if success:
			success_count += 1

		records.append(EpisodeRecord(observations=observations, actions=actions))

	return records, success_count


def build_windows(
	records: Sequence[EpisodeRecord],
	context_len: int,
	chunk_len: int,
	action_dim: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
	context_list: List[np.ndarray] = []
	target_action_list: List[np.ndarray] = []
	mask_list: List[np.ndarray] = []

	for ep in records:
		obs_seq = ep.observations
		act_seq = ep.actions
		t_max = len(obs_seq)
		for t in range(t_max):
			c = np.zeros((context_len, obs_seq[0].shape[0]), dtype=np.float32)
			for i in range(context_len):
				idx = t - context_len + 1 + i
				if idx < 0:
					c[i] = obs_seq[0]
				else:
					c[i] = obs_seq[idx]

			y = np.zeros((chunk_len, action_dim), dtype=np.float32)
			m = np.zeros((chunk_len, 1), dtype=np.float32)
			for k in range(chunk_len):
				idx = t + k
				if idx < len(act_seq):
					y[k] = act_seq[idx]
					m[k, 0] = 1.0

			context_list.append(c)
			target_action_list.append(y)
			mask_list.append(m)

	contexts = np.asarray(context_list, dtype=np.float32)
	targets = np.asarray(target_action_list, dtype=np.float32)
	masks = np.asarray(mask_list, dtype=np.float32)
	return contexts, targets, masks


class TransformerBlock(tf.keras.layers.Layer):
	def __init__(self, model_dim: int, num_heads: int, mlp_dim: int, dropout: float):
		super().__init__()
		self.ln1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
		self.self_attn = tf.keras.layers.MultiHeadAttention(
			num_heads=num_heads,
			key_dim=model_dim // num_heads,
			dropout=dropout,
		)
		self.ln2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
		self.mlp = tf.keras.Sequential(
			[
				tf.keras.layers.Dense(mlp_dim, activation="gelu"),
				tf.keras.layers.Dropout(dropout),
				tf.keras.layers.Dense(model_dim),
			]
		)

	def call(self, x: tf.Tensor, training: bool = False) -> tf.Tensor:
		y = self.ln1(x)
		x = x + self.self_attn(y, y, training=training)
		y = self.ln2(x)
		x = x + self.mlp(y, training=training)
		return x


class DecoderBlock(tf.keras.layers.Layer):
	def __init__(self, model_dim: int, num_heads: int, mlp_dim: int, dropout: float):
		super().__init__()
		self.ln1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
		self.self_attn = tf.keras.layers.MultiHeadAttention(
			num_heads=num_heads,
			key_dim=model_dim // num_heads,
			dropout=dropout,
		)
		self.ln2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
		self.cross_attn = tf.keras.layers.MultiHeadAttention(
			num_heads=num_heads,
			key_dim=model_dim // num_heads,
			dropout=dropout,
		)
		self.ln3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
		self.mlp = tf.keras.Sequential(
			[
				tf.keras.layers.Dense(mlp_dim, activation="gelu"),
				tf.keras.layers.Dropout(dropout),
				tf.keras.layers.Dense(model_dim),
			]
		)

	def call(self, inputs: Sequence[tf.Tensor], training: bool = False) -> tf.Tensor:
		q, memory = inputs
		x = q + self.self_attn(self.ln1(q), self.ln1(q), training=training)
		x = x + self.cross_attn(self.ln2(x), memory, training=training)
		x = x + self.mlp(self.ln3(x), training=training)
		return x


class ACTPolicy(tf.keras.Model):
	def __init__(
		self,
		obs_dim: int,
		action_dim: int,
		context_len: int,
		chunk_len: int,
		action_scale: np.ndarray,
		model_dim: int = 256,
		num_heads: int = 8,
		enc_layers: int = 3,
		dec_layers: int = 3,
		mlp_dim: int = 512,
		dropout: float = 0.1,
	):
		super().__init__()
		self.context_len = context_len
		self.chunk_len = chunk_len
		self.action_dim = action_dim

		self.obs_proj = tf.keras.layers.Dense(model_dim)
		self.enc_pos = self.add_weight(
			shape=(1, context_len, model_dim),
			initializer=tf.keras.initializers.RandomNormal(stddev=0.02),
			trainable=True,
			name="enc_pos_embedding",
		)
		self.encoder_blocks = []
		for i in range(enc_layers):
			blk = TransformerBlock(model_dim, num_heads, mlp_dim, dropout)
			setattr(self, f"enc_block_{i}", blk)
			self.encoder_blocks.append(blk)

		self.chunk_queries = self.add_weight(
			shape=(1, chunk_len, model_dim),
			initializer=tf.keras.initializers.RandomNormal(stddev=0.02),
			trainable=True,
			name="chunk_queries",
		)
		self.decoder_blocks = []
		for i in range(dec_layers):
			blk = DecoderBlock(model_dim, num_heads, mlp_dim, dropout)
			setattr(self, f"dec_block_{i}", blk)
			self.decoder_blocks.append(blk)

		self.ln_out = tf.keras.layers.LayerNormalization(epsilon=1e-6)
		self.action_head = tf.keras.layers.Dense(action_dim)
		self.action_scale = tf.constant(action_scale.reshape(1, 1, -1), dtype=tf.float32)

	def call(self, context: tf.Tensor, training: bool = False) -> tf.Tensor:
		x = self.obs_proj(context) + self.enc_pos
		for blk in self.encoder_blocks:
			x = blk(x, training=training)

		batch_size = tf.shape(context)[0]
		q = tf.repeat(self.chunk_queries, repeats=batch_size, axis=0)
		for blk in self.decoder_blocks:
			q = blk([q, x], training=training)

		q = self.ln_out(q)
		actions = tf.tanh(self.action_head(q)) * self.action_scale
		return actions


def train_act(
	model: tf.keras.Model,
	contexts: np.ndarray,
	targets: np.ndarray,
	masks: np.ndarray,
	batch_size: int,
	epochs: int,
	learning_rate: float,
	eval_freq: int = 0,
	eval_fn: Callable[[int], None] = None,
) -> List[float]:
	dataset = tf.data.Dataset.from_tensor_slices((contexts, targets, masks))
	dataset = dataset.shuffle(buffer_size=min(len(contexts), 20000)).batch(batch_size)
	dataset = dataset.prefetch(tf.data.AUTOTUNE)

	optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
	epoch_losses: List[float] = []

	for epoch in range(1, epochs + 1):
		losses = []
		for batch_context, batch_target, batch_mask in dataset:
			with tf.GradientTape() as tape:
				pred = model(batch_context, training=True)
				sq_err = tf.square(pred - batch_target)
				weighted = sq_err * batch_mask
				loss = tf.reduce_sum(weighted) / (tf.reduce_sum(batch_mask) + 1e-6)

			grads = tape.gradient(loss, model.trainable_variables)
			grads, _ = tf.clip_by_global_norm(grads, 10.0)
			optimizer.apply_gradients(zip(grads, model.trainable_variables))
			losses.append(float(loss.numpy()))

		epoch_loss = float(np.mean(losses))
		epoch_losses.append(epoch_loss)
		print(f"Epoch {epoch:03d}/{epochs} | ACT loss: {epoch_loss:.6f}")

		if wandb_is_active():
			wandb.log(
				{
					"train/epoch": epoch,
					"train/loss": epoch_loss,
					"train/lr": learning_rate,
				},
				step=epoch,
			)

		if eval_freq > 0 and epoch % eval_freq == 0 and eval_fn is not None:
			eval_fn(epoch)

	return epoch_losses


def rollout_act_episode(
	env: gym.Env,
	model: tf.keras.Model,
	context_len: int,
	chunk_len: int,
	max_steps: int,
	record_video: bool = False,
) -> Tuple[float, bool, List[np.ndarray]]:
	obs, _ = env.reset()
	obs_vec = extract_obs_vector(obs)
	obs_queue: Deque[np.ndarray] = collections.deque(maxlen=context_len)
	for _ in range(context_len):
		obs_queue.append(obs_vec.copy())

	pending_actions: List[np.ndarray] = []
	ep_reward = 0.0
	success = False
	frames: List[np.ndarray] = []

	if record_video:
		frame = env.render()
		if frame is not None:
			frames.append(frame)

	for _ in range(max_steps):
		if not pending_actions:
			context = np.stack(list(obs_queue), axis=0)[None, ...].astype(np.float32)
			action_chunk = model(context, training=False).numpy()[0]
			pending_actions = [a for a in action_chunk]

		action = pending_actions.pop(0)
		next_obs, reward, terminated, truncated, info = env.step(action)
		ep_reward += float(reward)
		success = success or bool(info.get("is_success", False))

		next_obs_vec = extract_obs_vector(next_obs)
		obs_queue.append(next_obs_vec)

		if record_video:
			frame = env.render()
			if frame is not None:
				frames.append(frame)

		if terminated or truncated:
			break

	return ep_reward, success, frames


def evaluate_act(
	env: gym.Env,
	model: tf.keras.Model,
	context_len: int,
	chunk_len: int,
	max_steps: int,
	num_episodes: int,
	log_video_to_wandb: bool = False,
) -> Tuple[float, float]:
	rewards: List[float] = []
	successes: List[float] = []
	for ep in range(num_episodes):
		record_video = log_video_to_wandb and ep == 0
		ep_reward, ep_success, ep_frames = rollout_act_episode(
			env,
			model,
			context_len=context_len,
			chunk_len=chunk_len,
			max_steps=max_steps,
			record_video=record_video,
		)
		rewards.append(ep_reward)
		successes.append(1.0 if ep_success else 0.0)

		if record_video and len(ep_frames) > 0 and wandb_is_active():
			video_array = np.transpose(np.array(ep_frames), (0, 3, 1, 2))
			wandb.log({"eval/video": wandb.Video(video_array, fps=20, format="mp4")})

	return float(np.mean(rewards)), float(np.mean(successes))


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="ACT for PandaPickAndPlace-v3")
	parser.add_argument("--env-id", type=str, default="PandaPickAndPlace-v3")
	parser.add_argument("--seed", type=int, default=42)
	parser.add_argument("--train-episodes", type=int, default=200)
	parser.add_argument("--max-steps", type=int, default=120)
	parser.add_argument("--expert-noise", type=float, default=0.05)
	parser.add_argument("--context-len", type=int, default=10)
	parser.add_argument("--chunk-len", type=int, default=12)
	parser.add_argument("--batch-size", type=int, default=256)
	parser.add_argument("--epochs", type=int, default=60)
	parser.add_argument("--lr", type=float, default=3e-4)
	parser.add_argument("--model-dim", type=int, default=256)
	parser.add_argument("--num-heads", type=int, default=8)
	parser.add_argument("--enc-layers", type=int, default=3)
	parser.add_argument("--dec-layers", type=int, default=3)
	parser.add_argument("--mlp-dim", type=int, default=512)
	parser.add_argument("--dropout", type=float, default=0.1)
	parser.add_argument("--eval-episodes", type=int, default=20)
	parser.add_argument("--eval-freq", type=int, default=20, help="Evaluate every N epochs (0 to disable)")
	parser.add_argument("--save", type=str, default="")
	parser.add_argument("--render", action="store_true")
	parser.add_argument("--wandb", action="store_true")
	parser.add_argument("--wandb-project", type=str, default="rl-book-act")
	parser.add_argument("--wandb-entity", type=str, default="")
	parser.add_argument("--wandb-run-name", type=str, default="")
	parser.add_argument("--gpu-mem-limit", type=int, default=10240, help="GPU memory limit in MB (default: 10240 for 10GB)")
	parser.add_argument("--wandb-video", action="store_true", help="Log eval video to wandb")
	return parser.parse_args()


def main() -> None:
	args = parse_args()
	set_seed(args.seed)
	register_panda_envs()
	if args.gpu_mem_limit > 0:
		limit_gpu_memory(args.gpu_mem_limit)

	# panda-gym does not accept render_mode=None.
	env_kwargs = {"max_episode_steps": args.max_steps}
	if args.render:
		env_kwargs["render_mode"] = "human"
	elif args.wandb_video:
		env_kwargs["render_mode"] = "rgb_array"
	env = gym.make(args.env_id, **env_kwargs)

	obs0, _ = env.reset(seed=args.seed)
	obs_dim = extract_obs_vector(obs0).shape[0]
	action_dim = env.action_space.shape[0]
	action_high = np.asarray(env.action_space.high, dtype=np.float32)

	print(f"Environment: {args.env_id}")
	print(f"Obs dim: {obs_dim}, Action dim: {action_dim}")

	if args.wandb:
		if wandb is None:
			raise ImportError("wandb is not installed. Install with: pip install wandb")

		run_name = args.wandb_run_name or f"act-{args.env_id}-seed{args.seed}"
		wandb.init(
			project=args.wandb_project,
			entity=args.wandb_entity or None,
			name=run_name,
			config={
				"algorithm": "ACT",
				"env_id": args.env_id,
				"seed": args.seed,
				"train_episodes": args.train_episodes,
				"max_steps": args.max_steps,
				"expert_noise": args.expert_noise,
				"context_len": args.context_len,
				"chunk_len": args.chunk_len,
				"batch_size": args.batch_size,
				"epochs": args.epochs,
				"learning_rate": args.lr,
				"model_dim": args.model_dim,
				"num_heads": args.num_heads,
				"enc_layers": args.enc_layers,
				"dec_layers": args.dec_layers,
				"mlp_dim": args.mlp_dim,
				"dropout": args.dropout,
				"eval_episodes": args.eval_episodes,
				"eval_freq": args.eval_freq,
			},
		)

	print("Collecting expert demonstrations...")
	records, expert_success = collect_expert_dataset(
		env,
		num_episodes=args.train_episodes,
		max_steps=args.max_steps,
		action_noise=args.expert_noise,
	)
	print(
		f"Collected {len(records)} episodes | "
		f"Expert success rate: {expert_success / max(1, len(records)):.3f}"
	)
	if wandb_is_active():
		wandb.log(
			{
				"data/expert_episodes": len(records),
				"data/expert_success_rate": expert_success / max(1, len(records)),
			},
			step=0,
		)

	contexts, targets, masks = build_windows(
		records,
		context_len=args.context_len,
		chunk_len=args.chunk_len,
		action_dim=action_dim,
	)
	print(f"Training samples: {len(contexts)}")
	if wandb_is_active():
		wandb.log(
			{
				"data/training_samples": len(contexts),
				"data/obs_dim": obs_dim,
				"data/action_dim": action_dim,
			},
			step=0,
		)

	model = ACTPolicy(
		obs_dim=obs_dim,
		action_dim=action_dim,
		context_len=args.context_len,
		chunk_len=args.chunk_len,
		action_scale=action_high,
		model_dim=args.model_dim,
		num_heads=args.num_heads,
		enc_layers=args.enc_layers,
		dec_layers=args.dec_layers,
		mlp_dim=args.mlp_dim,
		dropout=args.dropout,
	)

	# Build model weights once.
	_ = model(tf.zeros((1, args.context_len, obs_dim), dtype=tf.float32), training=False)
	model.summary()

	print("Training ACT policy...")
	
	def do_eval(epoch: int) -> None:
		print(f"Evaluating ACT policy at epoch {epoch}...")
		avg_reward, success_rate = evaluate_act(
			env,
			model,
			context_len=args.context_len,
			chunk_len=args.chunk_len,
			max_steps=args.max_steps,
			num_episodes=args.eval_episodes,
			log_video_to_wandb=args.wandb_video,
		)
		print(f"Epoch {epoch} | Evaluation reward: {avg_reward:.3f} | Success rate: {success_rate:.3f}")
		if wandb_is_active():
			wandb.log(
				{
					"eval/avg_reward": avg_reward,
					"eval/success_rate": success_rate,
				},
				step=epoch,
			)

	epoch_losses = train_act(
		model,
		contexts=contexts,
		targets=targets,
		masks=masks,
		batch_size=args.batch_size,
		epochs=args.epochs,
		learning_rate=args.lr,
		eval_freq=args.eval_freq,
		eval_fn=do_eval,
	)

	if args.eval_freq <= 0 or args.epochs % args.eval_freq != 0:
		do_eval(args.epochs)

	if wandb_is_active():
		wandb.log({"train/final_loss": epoch_losses[-1] if epoch_losses else float("nan")}, step=args.epochs)

	if args.save:
		model.save_weights(args.save)
		print(f"Saved ACT weights to: {args.save}")
		if wandb_is_active():
			wandb.log({"artifacts/saved_weights": args.save})

	env.close()
	if wandb_is_active():
		wandb.finish()


if __name__ == "__main__":
	main()
