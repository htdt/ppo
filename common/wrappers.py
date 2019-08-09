import torch
import gym
from gym.spaces.box import Box
from baselines.common.vec_env import VecEnvWrapper


class TransposeImage(gym.ObservationWrapper):
    def __init__(self, env=None, op=[2, 0, 1]):
        """
        Transpose observation space for images
        """
        super(TransposeImage, self).__init__(env)
        assert len(op) == 3, f"Error: Operation, {str(op)}, must be dim3"
        self.op = op
        obs_shape = self.observation_space.shape
        self.observation_space = Box(
            self.observation_space.low[0, 0, 0],
            self.observation_space.high[0, 0, 0], [
                obs_shape[self.op[0]], obs_shape[self.op[1]],
                obs_shape[self.op[2]]
            ],
            dtype=self.observation_space.dtype)

    def observation(self, ob):
        return ob.transpose(self.op[0], self.op[1], self.op[2])


class VecPyTorch(VecEnvWrapper):
    def reset(self):
        return torch.from_numpy(self.venv.reset())

    def step_async(self, actions):
        assert len(actions.shape) == 2
        self.venv.step_async(actions.squeeze(1).cpu().numpy())

    def step_wait(self):
        obs, reward, done, info = self.venv.step_wait()
        obs = torch.from_numpy(obs)
        reward = torch.from_numpy(reward).unsqueeze(dim=1)
        done = torch.tensor(done.tolist()).unsqueeze(dim=1)
        return obs, reward, done, info
