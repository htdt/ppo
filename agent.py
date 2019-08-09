from collections import defaultdict
from dataclasses import dataclass
import random
import torch

from common.optim import ParamOptim
from common.tools import log_grads
from model import ActorCritic


@dataclass
class Agent:
    model: ActorCritic
    optim: ParamOptim
    pi_clip: float
    epochs: int
    batch_size: int
    val_loss_k: float
    ent_k: float
    gamma: float
    gae_lambda: float

    def _gae(self, rollout):
        returns = torch.empty_like(rollout['vals'])
        gae = 0
        for i in reversed(range(returns.shape[0] - 1)):
            mask = rollout['masks'][i + 1]
            delta = rollout['rewards'][i] - rollout['vals'][i] +\
                self.gamma * rollout['vals'][i + 1] * mask
            gae = delta + self.gamma * self.gae_lambda * mask * gae
            returns[i] = gae + rollout['vals'][i]
        return returns

    def update(self, rollout):
        num_step, num_env = rollout['log_probs'].shape[:2]
        with torch.no_grad():
            rollout['vals'][-1] = self.model(rollout['obs'][-1])[1]
            returns = self._gae(rollout)

        adv = returns[:-1] - rollout['vals'][:-1]
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        logs, grads = defaultdict(list), defaultdict(list)

        for _ in range(self.epochs * num_step * num_env // self.batch_size):
            idx1d = random.sample(range(num_step * num_env), self.batch_size)
            idx = tuple(zip(*[(i % num_step, i // num_step) for i in idx1d]))

            dist, vals = self.model(rollout['obs'][idx])
            act = rollout['actions'][idx].squeeze(-1)
            log_probs = dist.log_prob(act).unsqueeze(-1)
            ent = dist.entropy().mean()

            old_lp = rollout['log_probs'][idx]
            ratio = torch.exp(log_probs - old_lp)
            surr1 = adv[idx] * ratio
            surr2 = adv[idx] * \
                torch.clamp(ratio, 1 - self.pi_clip, 1 + self.pi_clip)
            act_loss = -torch.min(surr1, surr2).mean()
            val_loss = (vals - returns[idx]).pow(2).mean()

            self.optim.step(-self.ent_k * ent + act_loss +
                            self.val_loss_k * val_loss)

            log_grads(self.model, grads)
            logs['ent'].append(ent)
            logs['clipfrac'].append(
                (torch.abs(ratio - 1) > self.pi_clip).float().mean())
            logs['loss/actor'].append(act_loss)
            logs['loss/critic'].append(val_loss)

        for name, val in grads.items():
            if '/max' in name:
                grads[name] = max(val)
            elif '/std' in name:
                grads[name] = sum(val) / (len(val) ** .5)
        return {
            'ent': torch.stack(logs['ent']).mean(),
            'clip/frac': torch.stack(logs['clipfrac']).mean(),
            'loss/actor': torch.stack(logs['loss/actor']).mean(),
            'loss/critic': torch.stack(logs['loss/critic']).mean(),
            **grads,
        }
