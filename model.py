from typing import List
import torch
import torch.nn as nn
from torch.distributions import Categorical
from common.conv import Conv
from common.cfg import find_checkpoint
from common.tools import init_ortho


class ActorCritic(nn.Module):
    def __init__(
        self,
        output_size: int,
        device: str,
        hidden_sizes: List[int],
        input_size: int = None,
        conv: Conv = None,
    ):
        super(ActorCritic, self).__init__()
        self.output_size = output_size
        self.device = device
        self.conv = conv

        if conv is not None:
            input_size = conv.output_size
        assert input_size is not None

        def with_relu(m):
            return nn.Sequential(init_ortho(m, 'relu'), nn.ReLU())

        self.fc = nn.Sequential(*[
            with_relu(nn.Linear(s_in, s_out))
            for s_in, s_out in zip(
                [input_size] + hidden_sizes[:-1], hidden_sizes)])
        input_size = hidden_sizes[-1]

        self.pi = init_ortho(nn.Linear(input_size, output_size), .01)
        self.val = init_ortho(nn.Linear(input_size, 1))

    def forward(self, x):
        if self.conv is not None:
            x = self.conv(x)
        if self.fc is not None:
            x = self.fc(x)
        return Categorical(logits=self.pi(x)), self.val(x)


def init_model(cfg, env, device, resume):
    obs_shape = env.observation_space.shape
    conv = Conv(**cfg['conv'], input_size=obs_shape) if 'conv' in cfg else None
    model = ActorCritic(
        hidden_sizes=cfg['model'].get('hidden_sizes'),
        output_size=env.action_space.n,
        conv=conv,
        device=device,
        input_size=obs_shape[0],
    ).train()
    model.to(device=device)

    if resume:
        n_start, fname = find_checkpoint(cfg)
        if fname:
            model.load_state_dict(torch.load(fname, map_location=device))
    else:
        n_start = 0

    return model, n_start
