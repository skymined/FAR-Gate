# ppo.py
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# ---------- 정책 + 가치 네트워크 ----------
class PolicyNet(nn.Module):
    def __init__(self, obs_dim, hidden=64):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
        )
        self.policy_head = nn.Linear(hidden, 2)   # protect / not
        self.value_head = nn.Linear(hidden, 1)

    def forward(self, x):
        h = self.body(x)
        logits = self.policy_head(h)
        value = self.value_head(h)
        return logits, value


# ---------- GAE (advantage 계산) ----------
def compute_gae(rewards, values, gamma=0.99, lam=0.95):
    adv = []
    gae = 0.0
    for t in reversed(range(len(rewards))):
        delta = rewards[t] + gamma * (values[t+1] if t+1 < len(values) else 0.0) - values[t]
        gae = delta + gamma * lam * gae
        adv.insert(0, gae)
    ret = [a + v for a, v in zip(adv, values)]
    return adv, ret


# ---------- PPO 업데이트 ----------
def ppo_update(policy, optimizer, batch, clip_eps=0.2, vf_coef=0.5, ent_coef=0.01):
    obs = torch.tensor(np.array(batch["obs"]), dtype=torch.float32)
    act = torch.tensor(batch["act"], dtype=torch.long)
    old_logp = torch.tensor(batch["logp"], dtype=torch.float32)
    ret = torch.tensor(batch["ret"], dtype=torch.float32)
    adv = torch.tensor(batch["adv"], dtype=torch.float32)

    logits, value = policy(obs)
    dist = torch.distributions.Categorical(logits=logits)
    logp = dist.log_prob(act)
    ratio = torch.exp(logp - old_logp)

    surr1 = ratio * adv
    surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * adv
    policy_loss = -torch.min(surr1, surr2).mean()
    value_loss = (ret - value.squeeze(-1)).pow(2).mean()
    entropy = dist.entropy().mean()

    loss = policy_loss + vf_coef * value_loss - ent_coef * entropy
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
