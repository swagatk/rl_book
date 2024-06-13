from collections import deque
import gymnasium as gym
from gym.spaces import Box
import numpy as np

class FrameStack(gym.Wrapper):
    """
    Wrapper that stacks observations from the environment into a single observation.

    This wrapper keeps a rolling buffer of the most recent frames and stacks them
    together as the new observation.
    """

    def __init__(self, env, num_stacked_frames):
        """
        Args:
          env: The environment to wrap.
          num_stacked_frames: The number of frames to stack.
        """
        super(FrameStack, self).__init__(env)

        self.num_stacked_frames = num_stacked_frames
        self.frames = deque([], maxlen=num_stacked_frames)
        obs_shape = env.observation_space.shape
        if len(obs_shape) == 2: # convert (H, W) to (H, W, D)
            obs_shape = obs_shape + (1,)

        # Modify the observation space to accommodate stacked frames
        self.observation_space = Box(
            low=0, high=255,
            shape=(obs_shape[0], obs_shape[1], obs_shape[2] * self.num_stacked_frames),
            dtype=self.env.observation_space.dtype
        )

    def reset(self):
        """
        Resets the environment and fills the frame buffer with initial observations.
        """
        observation = self.env.reset()[0]
        
        if len(np.shape(observation)) == 2: # convert (H, W) to (H, W, D)
            observation = np.expand_dims(observation, axis=2) 
        for _ in range(self.num_stacked_frames):
            self.frames.append(observation)
        return self._stack_frames()

    def step(self, action):
        """
        Steps through the environment and stacks the new observation with previous ones.
        """
        observation, reward, done, info, _ = self.env.step(action)
        if len(np.shape(observation)) == 2: # convert (H, W) to (H, W, D)
            observation = np.expand_dims(observation, axis=2) 
        self.frames.append(observation)
        return self._stack_frames(), reward, done, info

    def _stack_frames(self):
        """
        Stacks the frames from the buffer into a single observation.
        """
        return np.concatenate(self.frames, axis=2)

if __name__ == '__main__':

    env = gym.make('ALE/MsPacman-v5', obs_type="grayscale", render_mode='rgb_array')
    env = FrameStack(env, num_stacked_frames=4)
    print('Observation space shape:', env.observation_space.shape)
    print('state shape: ', state.shape)