from dataclasses import dataclass
import torch
import numpy as np
from baselines.common.vec_env.shmem_vec_env import ShmemVecEnv
from model import ActorCritic


@dataclass
class EnvRunner:
    envs: ShmemVecEnv
    model: ActorCritic
    rollout_size: int
    device: str
    ep_reward = []
    ep_len = []

    def get_logs(self):
        if len(self.ep_reward) >= self.envs.num_envs:
            res = {
                '_episode/reward': np.mean(self.ep_reward),
                '_episode/len': np.mean(self.ep_len),
            }
            self.ep_reward.clear(), self.ep_len.clear()
            return res
        else:
            return {}

    def __iter__(self):
        r, n = self.rollout_size, self.envs.num_envs

        def tensor(*shape, dtype=torch.float):
            return torch.empty(*shape, dtype=dtype, device=self.device)

        obs_shape = self.envs.observation_space.shape
        obs_dtype = torch.uint8 if len(obs_shape) == 4 else torch.float
        obs = tensor(r + 1, n, *obs_shape, dtype=obs_dtype)

        rewards = tensor(r, n, 1)
        vals = tensor(r + 1, n, 1)
        log_probs = tensor(r, n, 1)
        actions = tensor(r, n, 1, dtype=torch.long)
        masks = tensor(r + 1, n, 1)

        step = 0
        masks[0] = 1
        obs[0] = self.envs.reset()

        while True:
            with torch.no_grad():
                dist, vals[step] = self.model(obs[step])
                a = dist.sample()
                actions[step] = a.unsqueeze(-1)
                log_probs[step] = dist.log_prob(a).unsqueeze(-1)

            obs[step + 1], rewards[step], terms, infos =\
                self.envs.step(actions[step])
            masks[step + 1] = ~terms

            for i, info in enumerate(infos):
                if 'episode' in info.keys():
                    self.ep_reward.append(info['episode']['r'])
                    self.ep_len.append(info['episode']['l'])

            step = (step + 1) % self.rollout_size
            if step == 0:
                yield {'obs': obs,
                       'rewards': rewards,
                       'vals': vals,
                       'log_probs': log_probs,
                       'actions': actions,
                       'masks': masks,
                       }

                masks[0].copy_(masks[-1])
                obs[0].copy_(obs[-1])
