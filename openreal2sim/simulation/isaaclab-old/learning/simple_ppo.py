# ppo_utils.py
from __future__ import annotations
from pathlib import Path
from typing import Tuple, Iterable
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

# ========================= Actor-Critic =========================
class ActorCritic(nn.Module):
    """Simple MLP actor-critic. Actor outputs mean in pre-squash space; action = tanh(N(mu, sigma))."""
    def __init__(self, obs_dim: int, act_dim: int, hidden_sizes: Tuple[int, ...] = (256, 256)):
        super().__init__()
        layers = []
        last = obs_dim
        for hs in hidden_sizes:
            layers += [nn.Linear(last, hs), nn.ReLU(inplace=True)]
            last = hs
        self.shared = nn.Sequential(*layers)
        self.mu_head = nn.Linear(last, act_dim)                       # pre-squash mean
        self.log_std = nn.Parameter(torch.full((act_dim,), -0.5))     # trainable log_std
        self.v_head  = nn.Linear(last, 1)

    def forward(self, obs: torch.Tensor):
        """obs: (B, obs_dim) -> mu: (B, act_dim), std: (act_dim,), v: (B,)"""
        x = self.shared(obs)
        mu = self.mu_head(x)
        v  = self.v_head(x).squeeze(-1)
        std = torch.exp(self.log_std)
        return mu, std, v

    @staticmethod
    def sample_tanh_gaussian(mu: torch.Tensor, std: torch.Tensor):
        """Return action in [-1,1], log_prob with tanh correction, and pre-squash sample u."""
        dist = torch.distributions.Normal(mu, std)
        u = dist.rsample()  # reparameterized
        a = torch.tanh(u)
        # log|det(Jacobian of tanh)| = sum log(1 - tanh(u)^2)
        log_prob = dist.log_prob(u) - torch.log(1.0 - a.pow(2) + 1e-6)
        return a, log_prob.sum(-1), u

    @staticmethod
    def log_prob_tanh_gaussian(mu: torch.Tensor, std: torch.Tensor, a: torch.Tensor):
        """Compute log_prob(a) for tanh(N(mu,std)) using atanh(a)."""
        a = a.clamp(min=-0.999999, max=0.999999)
        atanh_a = 0.5 * (torch.log1p(a + 1e-6) - torch.log1p(-a + 1e-6))
        dist = torch.distributions.Normal(mu, std)
        log_prob = dist.log_prob(atanh_a) - torch.log(1.0 - a.pow(2) + 1e-6)
        return log_prob.sum(-1)


# ========================= Rollout Buffer =========================
class RolloutBuffer:
    """On-policy PPO rollout buffer with fixed horizon (all tensors on a single device)."""
    def __init__(self, horizon: int, num_envs: int, obs_dim: int, act_dim: int, device: torch.device):
        self.h, self.n, self.d, self.k = horizon, num_envs, obs_dim, act_dim
        self.device = device
        self.reset()

    def reset(self):
        H, N, D, K, dev = self.h, self.n, self.d, self.k, self.device
        self.obs      = torch.zeros(H, N, D, device=dev, dtype=torch.float32)
        self.actions  = torch.zeros(H, N, K, device=dev, dtype=torch.float32)
        self.logp     = torch.zeros(H, N, device=dev, dtype=torch.float32)
        self.rewards  = torch.zeros(H, N, device=dev, dtype=torch.float32)
        self.dones    = torch.zeros(H, N, device=dev, dtype=torch.bool)
        self.values   = torch.zeros(H, N, device=dev, dtype=torch.float32)
        self.timeouts = torch.zeros(H, N, device=dev, dtype=torch.bool)
        self.ptr = 0

    def add(self, *, obs, act, logp, rew, done, val, t_out):
        """Store DETACHED copies to keep env and rollout outside the grad graph."""
        i = self.ptr
        self.obs[i]      = obs.detach()
        self.actions[i]  = act.detach()
        self.logp[i]     = logp.detach()
        self.rewards[i]  = rew.detach().squeeze(-1)  # allow (B,1) or (B,)
        self.dones[i]    = done.detach().bool()
        self.values[i]   = val.detach()
        self.timeouts[i] = t_out.detach().bool()
        self.ptr += 1

    @torch.no_grad()  # make GAE graph-free
    def compute_gae(self, last_value: torch.Tensor, gamma=0.99, lam=0.95):
        """last_value: (N,)"""
        H, N = self.h, self.n
        adv = torch.zeros(H, N, device=self.device, dtype=torch.float32)
        last_gae = torch.zeros(N, device=self.device, dtype=torch.float32)
        for t in reversed(range(H)):
            if t == H - 1:
                next_nonterminal = (~self.dones[t] | self.timeouts[t]).float()
                next_values = last_value
            else:
                next_nonterminal = (~self.dones[t + 1] | self.timeouts[t + 1]).float()
                next_values = self.values[t + 1]
            delta = self.rewards[t] + gamma * next_values * next_nonterminal - self.values[t]
            last_gae = delta + gamma * lam * next_nonterminal * last_gae
            adv[t] = last_gae
        ret = adv + self.values
        return adv, ret

    def iterate_minibatches(self, batch_size: int, num_epochs: int) -> Iterable:
        H, N = self.h, self.n
        total = H * N
        obs     = self.obs.view(total, -1)
        acts    = self.actions.view(total, -1)
        logp    = self.logp.view(total)
        values  = self.values.view(total)

        def idx_batches():
            idx = torch.randperm(total, device=self.device)
            for i in range(0, total, batch_size):
                j = idx[i : i + batch_size]
                yield j

        for _ in range(num_epochs):
            for j in idx_batches():
                yield j, obs[j], acts[j], logp[j], values[j]


# ========================= Helpers =========================
def _as_tensor(x, device: torch.device, dtype: torch.dtype | None = None) -> torch.Tensor:
    """Convert numpy/torch to torch on device with optional dtype, avoiding copies when possible."""
    if isinstance(x, torch.Tensor):
        if dtype is not None and x.dtype != dtype:
            x = x.to(dtype)
        return x.to(device)
    return torch.as_tensor(x, device=device, dtype=dtype)


@torch.no_grad()
def _reset_env(env, device: torch.device):
    """Support env.reset() that returns torch or numpy."""
    rst = env.reset()
    obs = rst["obs"]
    return _as_tensor(obs, device=device, dtype=torch.float32)


def _choose_minibatch_size(total_batch: int, num_envs: int) -> int:
    """Heuristic minibatch sizing that divides total_batch."""
    mb = max(num_envs * 2, 32)
    while total_batch % mb != 0 and mb > 8:
        mb //= 2
    if total_batch % mb != 0:
        mb = num_envs
    return mb


@torch.no_grad()
def _partial_reset_and_merge_obs(env, next_obs: torch.Tensor, done_t: torch.Tensor, device: torch.device):
    """
    If any env is done:
      1) Try env.reset_done(done_mask) which returns {"obs": (M,D)} or (N,D)
      2) Else try env.reset(env_ids=idxs)
      3) Else fallback to global reset()
    Then merge the reset obs into next_obs only for those done envs and return merged tensor.
    """
    if not bool(done_t.any()):
        return next_obs  # nothing to do

    idxs = torch.nonzero(done_t, as_tuple=False).squeeze(-1)  # (M,)
    M = int(idxs.numel())
    if M == 0:
        return next_obs

    rst = env.reset(env_ids=idxs)
    new_obs = _as_tensor(rst["obs"], device=device, dtype=torch.float32)
    if new_obs.shape[0] == next_obs.shape[0]:
        return new_obs
    merged = next_obs.clone()
    merged[idxs] = new_obs
    return merged

# ========================= PPO Train =========================
def ppo_train(
    *,
    env,
    net: ActorCritic,
    optimizer: optim.Optimizer,
    device: torch.device,
    total_epochs: int,
    horizon: int,
    mini_epochs: int,
    batch_size_hint: int,
    clip_eps: float,
    value_coef: float,
    entropy_coef: float,
    gamma: float,
    gae_lambda: float,
    run_dir: Path,
    save_every: int = 50,
):
    """
    Generic PPO training loop.
    Assumes:
      - env.reset() -> {"obs": torch or numpy, shape (N,D)}
      - env.step(action) ->
            ({"obs": torch/numpy (N,D)}, reward (N,1)/(N,), done (N,), {"time_outs": (N,)})
      - Partial reset:
            env.reset_done(done_mask)   OR   env.reset(env_ids=idxs)   OR   fallback global reset()
    """
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_best = run_dir / "best.pt"

    num_envs = getattr(env, "num_envs", None)
    if num_envs is None:
        raise ValueError("env.num_envs is required for batch sizing.")
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    writer = SummaryWriter(log_dir=run_dir)

    print(f"[INFO] PPO training: epochs={total_epochs}, horizon={horizon}, num_envs={num_envs}, "
          f"obs_dim={obs_dim}, act_dim={act_dim}")
    buffer = RolloutBuffer(horizon=horizon, num_envs=num_envs, obs_dim=obs_dim, act_dim=act_dim, device=device)

    total_batch = horizon * num_envs if batch_size_hint <= 0 else int(batch_size_hint)
    minibatch_size = _choose_minibatch_size(total_batch, num_envs)

    # initial reset
    obs = _reset_env(env, device)
    best_mean_return = -1e9

    for epoch in range(1, total_epochs + 1):
        buffer.reset()

        # ===== rollout =====
        for t in range(horizon):
            mu, std, v = net(obs)
            act, logp, _ = net.sample_tanh_gaussian(mu, std)  # act in [-1,1], (N,K)

            # IMPORTANT: do not let env/sim see gradients
            act_no_grad = act.detach()

            step_out = env.step(act_no_grad)
            next_obs_any, reward_any, done_any, infos = step_out[0]["obs"], step_out[1], step_out[2], step_out[3]

            next_obs = _as_tensor(next_obs_any, device=device, dtype=torch.float32)
            rew_t    = _as_tensor(reward_any,  device=device, dtype=torch.float32)  # (N,1) or (N,)
            done_t   = _as_tensor(done_any,    device=device, dtype=torch.bool)     # (N,)
            t_out    = _as_tensor(infos.get("time_outs", torch.zeros_like(done_t)), device=device, dtype=torch.bool)

            buffer.add(obs=obs, act=act_no_grad, logp=logp, rew=rew_t, done=done_t, val=v, t_out=t_out)

            # partial reset (merge only done envs' obs)
            next_obs = _partial_reset_and_merge_obs(env, next_obs, done_t, device)

            obs = next_obs

        # ===== bootstrap value =====
        _, _, last_v = net(obs)  # (N,)

        # ===== GAE / returns (graph-free) =====
        adv, ret = buffer.compute_gae(last_value=last_v.detach(), gamma=gamma, lam=gae_lambda)

        # normalize advantage
        adv_flat = adv.view(-1)
        ret_flat = ret.view(-1)
        adv_flat = (adv_flat - torch.mean(adv_flat)) / (torch.std(adv_flat) + 1e-8)

        # ===== PPO updates =====
        policy_loss_epoch = 0.0
        value_loss_epoch  = 0.0
        entropy_epoch     = 0.0
        num_updates       = 0

        for idx, obs_b, act_b, old_logp_b, old_v_b in buffer.iterate_minibatches(minibatch_size, mini_epochs):
            adv_b = adv_flat[idx]
            ret_b = ret_flat[idx]

            mu_b, std_b, v_b = net(obs_b)
            new_logp_b = net.log_prob_tanh_gaussian(mu_b, std_b, act_b)
            ratio = torch.exp(new_logp_b - old_logp_b)

            surrogate1 = ratio * adv_b
            surrogate2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * adv_b
            policy_loss = -torch.min(surrogate1, surrogate2).mean()

            value_clipped = old_v_b + (v_b - old_v_b).clamp(-clip_eps, clip_eps)
            value_losses = (v_b - ret_b).pow(2)
            value_losses_clipped = (value_clipped - ret_b).pow(2)
            value_loss = 0.5 * torch.max(value_losses, value_losses_clipped).mean()

            base_dist = torch.distributions.Normal(mu_b, std_b)
            entropy = base_dist.entropy().sum(-1).mean()

            loss = policy_loss + value_coef * value_loss - entropy_coef * entropy

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 0.5)
            optimizer.step()

            policy_loss_epoch += float(policy_loss.detach())
            value_loss_epoch  += float(value_loss.detach())
            entropy_epoch     += float(entropy.detach())
            num_updates       += 1

        mean_return = float(ret.mean().detach())
        print(f"[E{epoch:04d}] return={mean_return:7.3f}  "
              f"pi={policy_loss_epoch/max(1,num_updates):7.4f}  "
              f"v={value_loss_epoch/max(1,num_updates):7.4f}  "
              f"H={entropy_epoch/max(1,num_updates):7.4f}")

        # Value head quality: explained variance on flattened batch
        ev = float(1.0 - torch.var(ret_flat - buffer.values.view(-1)) / (torch.var(ret_flat) + 1e-12))

        writer.add_scalar("return_mean", mean_return, epoch)
        writer.add_scalar("policy_loss", policy_loss_epoch / max(1,num_updates), epoch)
        writer.add_scalar("value_loss", value_loss_epoch / max(1,num_updates), epoch)
        writer.add_scalar("entropy", entropy_epoch / max(1,num_updates), epoch)
        writer.add_scalar("explained_variance", ev, epoch)

        # save best & periodic
        if mean_return > best_mean_return:
            best_mean_return = mean_return
            torch.save({"model": net.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "epoch": epoch,
                        "best_return": best_mean_return},
                       ckpt_best)
        if (save_every > 0) and (epoch % save_every == 0):
            ckpt_path = run_dir / f"epoch_{epoch:04d}_ret_{mean_return:.2f}.pt"
            torch.save({"model": net.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "epoch": epoch,
                        "return": mean_return},
                       ckpt_path)

    return {"best_mean_return": best_mean_return, "ckpt_best": ckpt_best}
