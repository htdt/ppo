import sys
import time
from tqdm import trange
import torch
from common.make_env import make_otc
from common.tools import load_cfg
from ppo.from_config import init_model


def render(cfg_name, steps, seed):
    cfg = load_cfg(cfg_name, 'ppo')
    cfg['env']['num'] = 1
    cfg['env']['seed'] = seed
    env = make_otc(**cfg['env'], show=True)
    model, n_start = init_model(cfg, env, 'cpu', resume=True)
    assert n_start > 0
    model.eval()
    print(f'running {n_start}')

    fr, num_env = cfg['agent']['frames'], cfg['env']['num']
    hx_zero, hx_size = cfg['model']['hx_zero'], cfg['model']['rnn_size']
    obs = torch.zeros(fr, num_env, *env.observation_space.shape)
    obs_vec = torch.zeros(fr, num_env, env.observation_space_vec.shape[0])
    rewards = torch.zeros(fr, num_env, 1)
    actions = torch.zeros(fr, num_env, 1, dtype=torch.long)
    masks = torch.zeros(fr, num_env, 1)
    if not hx_zero:
        hx = torch.zeros(fr, num_env, hx_size)

    masks[-1] = 1
    obs[-1], obs_vec[-1] = env.reset()

    for n_iter in trange(steps):
        with torch.no_grad():
            dist, _, hx_next = model(obs, obs_vec, actions, rewards, masks,
                                     None if hx_zero else hx[0])
            a = dist.sample()

        rewards[:-1].copy_(rewards[1:])
        actions[:-1].copy_(actions[1:])
        masks[:-1].copy_(masks[1:])
        obs[:-1].copy_(obs[1:])
        obs_vec[:-1].copy_(obs_vec[1:])
        if not hx_zero:
            hx[:-1].copy_(hx[1:])
            hx[-1] = hx_next

        actions[-1] = a.unsqueeze(-1)
        (obs[-1], obs_vec[-1]), rewards[-1], terms, infos =\
            env.step(actions[-1])
        masks[-1] = 1 - terms

        floor_up = obs_vec[-1, :, 2] > obs_vec[-2, :, 2]
        got_key = obs_vec[-1, :, 0] > obs_vec[-2, :, 0]
        masks[-1] *= (1 - floor_up).unsqueeze(-1).float()
        rewards[-1] += got_key.unsqueeze(-1).float()
        if rewards[-1].item() != 0:
            print(rewards[-1].item())
        time.sleep(1/30)


if __name__ == '__main__':
    assert len(sys.argv) == 4, 'required: config name, steps, seed'
    render(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]))
