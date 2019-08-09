import gym
from baselines import bench
from baselines.common.atari_wrappers import make_atari, wrap_deepmind
from baselines.common.vec_env.dummy_vec_env import DummyVecEnv
from baselines.common.vec_env.shmem_vec_env import ShmemVecEnv
from common.wrappers import TransposeImage, VecPyTorch


def make_vec_envs(name, num, seed=0):
    def make_env(rank):
        def _thunk():
            env = gym.make(name)
            is_atari = hasattr(gym.envs, 'atari') and isinstance(
                env.unwrapped, gym.envs.atari.atari_env.AtariEnv)
            if is_atari:
                env = make_atari(name)

            env.seed(seed + rank)
            env = bench.Monitor(env, None)
            if is_atari:
                env = wrap_deepmind(env)
            obs_shape = env.observation_space.shape
            if len(obs_shape) == 3 and obs_shape[2] in [1, 3]:
                env = TransposeImage(env, op=[2, 0, 1])
            return env
        return _thunk

    envs = [make_env(i) for i in range(num)]
    envs = DummyVecEnv(envs) if num == 1 else ShmemVecEnv(envs, context='fork')
    envs = VecPyTorch(envs)
    return envs
