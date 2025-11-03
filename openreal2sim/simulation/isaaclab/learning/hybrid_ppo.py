# learning/ppo_hybrid.py
# -*- coding: utf-8 -*-
"""
Hybrid PPO: Categorical (proposal index at t=0) + Gaussian (offset at t=0) + Gaussian (residual for t>0).
Action sent to env is always concatenated as:
    [ residual_6 | proposal_onehot_P | offset_6 ]
Env consumes:
    - t == 0: uses (proposal_onehot_P, offset_6); ignores residual_6
    - t > 0 : uses residual_6; ignores the rest
"""

from __future__ import annotations
from pathlib import Path
from typing import Tuple, Iterable, Dict

import torch
import torch.nn as nn
import torch.optim as optim


# ========================= Hybrid Actor-Critic =========================
class HybridActorCritic(nn.Module):
    """
    Policy head structure:
      - proposal_logits: (B, P)          [Categorical at t=0]
      - offset_mu/std:   (B, P, 6)        [Gaussian at t=0, choose row by sampled idx]
      - residual_mu/std: (B, 6)           [Gaussian at t>0]
      - value:           (B,)
    """
    def __init__(self, obs_dim: int, num_proposals: int, hidden_sizes: Tuple[int, ...] = (256, 256)):
        super().__init__()
        self.P = int(num_proposals)

        # Shared trunk
        layers = []
        last = obs_dim
        for hs in hidden_sizes:
            layers += [nn.Linear(last, hs), nn.ReLU(inplace=True)]
            last = hs
        self.shared = nn.Sequential(*layers)

        # Heads
        self.proposal_head = nn.Linear(last, self.P)         # logits for Categorical
        self.offset_mu     = nn.Linear(last, self.P * 6)     # per-proposal 6D mean
        self.offset_logstd = nn.Parameter(torch.full((self.P, 6), -0.5))  # learnable, proposal-wise
        self.residual_mu   = nn.Linear(last, 6)              # 6D mean for residual
        self.residual_logstd = nn.Parameter(torch.full((6,), -0.5))       # learnable

        self.v_head        = nn.Linear(last, 1)

    def forward(self, obs: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Returns a dict of all distribution params and value.
        Shapes:
          - proposal_logits: (B,P)
          - offset_mu:       (B,P,6)
          - offset_std:      (P,6)  (broadcast on batch)
          - residual_mu:     (B,6)
          - residual_std:    (6,)   (broadcast on batch)
          - value:           (B,)
        """
        x = self.shared(obs)
        proposal_logits = self.proposal_head(x)                              # (B,P)
        offset_mu = self.offset_mu(x).view(-1, self.P, 6)                    # (B,P,6)
        offset_std = torch.exp(self.offset_logstd)                           # (P,6)
        residual_mu = self.residual_mu(x)                                    # (B,6)
        residual_std = torch.exp(self.residual_logstd)                       # (6,)
        value = self.v_head(x).squeeze(-1)
        return {
            "proposal_logits": proposal_logits,
            "offset_mu": offset_mu,
            "offset_std": offset_std,
            "residual_mu": residual_mu,
            "residual_std": residual_std,
            "value": value,
        }

    @staticmethod
    def _one_hot(indices: torch.Tensor, P: int) -> torch.Tensor:
        """indices: (B,), return onehot (B,P) float32."""
        B = indices.shape[0]
        onehot = torch.zeros(B, P, dtype=torch.float32, device=indices.device)
        onehot.scatter_(1, indices.view(-1, 1), 1.0)
        return onehot

    def act_sample(self, obs: torch.Tensor, is_step0: bool) -> Dict[str, torch.Tensor]:
        """
        Sample action and return everything needed for PPO rollout at this step.
        At t=0: sample idx ~ Categorical, offset ~ N(mu[idx], std[idx]), build env_action.
        At t>0: sample residual ~ N(mu, std), build env_action with zeros for the rest.
        Returns dict with keys:
          - env_action: (B, 6 + P + 6)
          - logp:       (B,)
          - entropy:    (B,)
          - value:      (B,)
          - cache: any tensors needed for debugging (optional)
        """
        outs = self.forward(obs)
        P = outs["proposal_logits"].shape[1]
        B = obs.shape[0]

        if is_step0:
            # Categorical over proposals
            cat = torch.distributions.Categorical(logits=outs["proposal_logits"])
            idx = cat.sample()  # (B,)
            onehot = self._one_hot(idx, P)  # (B,P)

            # Gather offset distribution of the selected proposal
            mu_sel = outs["offset_mu"][torch.arange(B, device=obs.device), idx, :]           # (B,6)
            std_sel = outs["offset_std"][idx, :] if outs["offset_std"].dim() == 2 else outs["offset_std"]  # (B?,6)
            # outs["offset_std"] shape is (P,6), need to gather then expand to (B,6)
            std_sel = outs["offset_std"][idx, :]  # (B,6)

            gauss_off = torch.distributions.Normal(mu_sel, std_sel)
            u_off = gauss_off.rsample()  # (B,6)
            u_off_tanh = torch.tanh(u_off)  # keep action in [-1,1] space for env’s clamp
            # Tanh correction
            logp_off = gauss_off.log_prob(u_off) - torch.log(1.0 - u_off_tanh.pow(2) + 1e-6)
            logp_off = logp_off.sum(-1)  # (B,)

            logp = cat.log_prob(idx) + logp_off

            # Entropy (encourage exploration at t=0): categorical + offset gaussian entropy
            ent = cat.entropy() + gauss_off.entropy().sum(-1)

            # Residual part zero at t=0 (env ignores it anyway)
            residual = torch.zeros(B, 6, dtype=torch.float32, device=obs.device)

            # Build env action: [residual_6 | onehot_P | offset6]
            env_action = torch.cat([residual, onehot, u_off_tanh], dim=-1)  # (B, 6+P+6)

            return {
                "env_action": env_action,
                "logp": logp,
                "entropy": ent,
                "value": outs["value"],
            }
        else:
            # Only residual Gaussian matters
            gauss_res = torch.distributions.Normal(outs["residual_mu"], outs["residual_std"])
            u_res = gauss_res.rsample()          # (B,6)
            a_res = torch.tanh(u_res)            # (B,6)
            logp_res = gauss_res.log_prob(u_res) - torch.log(1.0 - a_res.pow(2) + 1e-6)
            logp_res = logp_res.sum(-1)          # (B,)
            ent = gauss_res.entropy().sum(-1)    # (B,)

            # proposal onehot & offset set to zero placeholders
            P = outs["proposal_logits"].shape[1]
            onehot = torch.zeros(B, P, dtype=torch.float32, device=obs.device)
            off6   = torch.zeros(B, 6, dtype=torch.float32, device=obs.device)

            env_action = torch.cat([a_res, onehot, off6], dim=-1)  # (B,6+P+6)
            return {
                "env_action": env_action,
                "logp": logp_res,
                "entropy": ent,
                "value": outs["value"],
            }

    def log_prob_recompute(self, obs: torch.Tensor, env_action: torch.Tensor, is_step0: bool) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Recompute log_prob and entropy under NEW network params for PPO update.
        env_action: (B, 6+P+6) = [res6 | onehot_P | off6]
        Returns (logp, entropy) both (B,)
        """
        outs = self.forward(obs)
        B = env_action.shape[0]
        P = outs["proposal_logits"].shape[1]

        res6  = env_action[:, :6]
        onehot = env_action[:, 6:6+P]
        off6 = env_action[:, 6+P:]

        if is_step0:
            # Recover idx from onehot
            idx = onehot.argmax(dim=-1)  # (B,)

            # logp for categorical
            cat = torch.distributions.Categorical(logits=outs["proposal_logits"])
            logp_cat = cat.log_prob(idx)  # (B,)

            # logp for selected offset
            mu_sel  = outs["offset_mu"][torch.arange(B, device=obs.device), idx, :]  # (B,6)
            std_sel = outs["offset_std"][idx, :]                                     # (B,6)
            gauss_off = torch.distributions.Normal(mu_sel, std_sel)
            # Convert bounded action back to pre-squash space via atanh
            a = off6.clamp(-0.999999, 0.999999)
            atanh_a = 0.5 * (torch.log1p(a + 1e-6) - torch.log1p(-a + 1e-6))
            logp_off = gauss_off.log_prob(atanh_a) - torch.log(1.0 - a.pow(2) + 1e-6)
            logp_off = logp_off.sum(-1)

            logp = logp_cat + logp_off
            ent = cat.entropy() + gauss_off.entropy().sum(-1)
            return logp, ent
        else:
            gauss_res = torch.distributions.Normal(outs["residual_mu"], outs["residual_std"])
            a = res6.clamp(-0.999999, 0.999999)
            atanh_a = 0.5 * (torch.log1p(a + 1e-6) - torch.log1p(-a + 1e-6))
            logp_res = gauss_res.log_prob(atanh_a) - torch.log(1.0 - a.pow(2) + 1e-6)
            logp_res = logp_res.sum(-1)
            ent = gauss_res.entropy().sum(-1)
            return logp_res, ent


# ========================= Rollout Buffer (Hybrid) =========================
class HybridRolloutBuffer:
    """
    Stores per-timestep data for PPO with fixed horizon.
    We keep the merged env_action (for env stepping and PPO recompute),
    together with scalar logp/value/reward/done/timeout.
    """
    def __init__(self, horizon: int, num_envs: int, obs_dim: int, act_dim: int, device: torch.device):
        self.h, self.n, self.d, self.k = horizon, num_envs, obs_dim, act_dim
        self.device = device
        self.reset()

    def reset(self):
        H, N, D, K, dev = self.h, self.n, self.d, self.k, self.device
        self.obs      = torch.zeros(H, N, D, device=dev, dtype=torch.float32)
        self.actions  = torch.zeros(H, N, K, device=dev, dtype=torch.float32)  # [res6|1hotP|off6]
        self.logp     = torch.zeros(H, N, device=dev, dtype=torch.float32)
        self.rewards  = torch.zeros(H, N, device=dev, dtype=torch.float32)
        self.dones    = torch.zeros(H, N, device=dev, dtype=torch.bool)
        self.values   = torch.zeros(H, N, device=dev, dtype=torch.float32)
        self.timeouts = torch.zeros(H, N, device=dev, dtype=torch.bool)
        self.ptr = 0

    def add(self, *, obs, env_action, logp, rew, done, val, t_out):
        i = self.ptr
        self.obs[i]      = obs.detach()
        self.actions[i]  = env_action.detach()
        self.logp[i]     = logp.detach()
        self.rewards[i]  = rew.detach().squeeze(-1)
        self.dones[i]    = done.detach().bool()
        self.values[i]   = val.detach()
        self.timeouts[i] = t_out.detach().bool()
        self.ptr += 1

    @torch.no_grad()
    def compute_gae(self, last_value: torch.Tensor, gamma=0.99, lam=0.95):
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
        obs   = self.obs.view(total, -1)
        acts  = self.actions.view(total, -1)
        logp  = self.logp.view(total)
        values= self.values.view(total)

        def idx_batches():
            idx = torch.randperm(total, device=self.device)
            for i in range(0, total, batch_size):
                j = idx[i: i + batch_size]
                yield j

        for _ in range(num_epochs):
            for j in idx_batches():
                yield j, obs[j], acts[j], logp[j], values[j]


# ========================= Helpers =========================
def _as_tensor(x, device: torch.device, dtype: torch.dtype | None = None) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        if dtype is not None and x.dtype != dtype:
            x = x.to(dtype)
        return x.to(device)
    return torch.as_tensor(x, device=device, dtype=dtype)

@torch.no_grad()
def _reset_env(env, device: torch.device):
    rst = env.reset()
    obs = rst["obs"]
    return _as_tensor(obs, device=device, dtype=torch.float32)

def _choose_minibatch_size(total_batch: int, num_envs: int) -> int:
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

# ========================= PPO Train (Hybrid) =========================
def ppo_train_hybrid(
    *,
    env,
    net: HybridActorCritic,
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
    Hybrid PPO training loop:
      - t=0 uses Categorical(index) + Gaussian(offset)
      - t>0 uses Gaussian(residual)
    Assumptions about env:
      - action_space.shape[0] == (6 + P + 6)
      - step() consumes the onehot & offset at t=0, and residual at t>0
    """
    from torch.utils.tensorboard import SummaryWriter

    print("#####################################horizon =", horizon)

    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_best = run_dir / "best.pt"

    num_envs = getattr(env, "num_envs", None)
    if num_envs is None:
        raise ValueError("env.num_envs is required for batch sizing.")
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]  # 6 + P + 6

    writer = SummaryWriter(log_dir=run_dir)

    print(f"[INFO][Hybrid PPO] epochs={total_epochs}, horizon={horizon}, num_envs={num_envs}, "
          f"obs_dim={obs_dim}, act_dim={act_dim}")
    buffer = HybridRolloutBuffer(horizon=horizon, num_envs=num_envs, obs_dim=obs_dim, act_dim=act_dim, device=device)

    total_batch = horizon * num_envs if batch_size_hint <= 0 else int(batch_size_hint)
    minibatch_size = _choose_minibatch_size(total_batch, num_envs)

    # initial reset
    obs = _reset_env(env, device)
    best_mean_return = -1e9

    for epoch in range(1, total_epochs + 1):
        buffer.reset()

        ep_return_per_env = torch.zeros(num_envs, device=device, dtype=torch.float32)
        finished_episode_returns = []  # Python list[float]
        ep_track_return_per_env = torch.zeros(num_envs, device=device, dtype=torch.float32)
        finished_episode_track_returns = []  # Python list[float]

        # ===== rollout =====
        for t in range(horizon):
            is_step0 = (t == 0)  # because your env uses fixed horizon episodes
            out = net.act_sample(obs, is_step0=is_step0)
            env_action = out["env_action"].detach()
            logp = out["logp"].detach()
            val  = out["value"].detach()

            step_out = env.step(env_action)  # action shape (N, 6+P+6)
            next_obs_any, reward_any, done_any, infos = step_out[0]["obs"], step_out[1], step_out[2], step_out[3]

            next_obs = _as_tensor(next_obs_any, device=device, dtype=torch.float32)
            rew_t    = _as_tensor(reward_any,  device=device, dtype=torch.float32)
            done_t   = _as_tensor(done_any,    device=device, dtype=torch.bool)
            t_out    = _as_tensor(infos.get("time_outs", torch.zeros_like(done_t)), device=device, dtype=torch.bool)

            buffer.add(obs=obs, env_action=env_action, logp=logp, rew=rew_t, done=done_t, val=val, t_out=t_out)

            # episode return bookkeeping
            r_track_t = _as_tensor(infos["r_track"], device=device, dtype=torch.float32).view_as(rew_t)
            ep_return_per_env += rew_t.squeeze(1)                  # (N,)
            ep_track_return_per_env += r_track_t.squeeze(1)        # (N,)

            done_or_timeout = (done_t | t_out)          # (N,)
            if bool(done_or_timeout.any()):
                finished_episode_returns.extend(
                    ep_return_per_env[done_or_timeout].detach().cpu().tolist()
                )
                finished_episode_track_returns.extend(
                    ep_track_return_per_env[done_or_timeout].detach().cpu().tolist()
                )
                ep_return_per_env[done_or_timeout] = 0.0
                ep_track_return_per_env[done_or_timeout] = 0.0

            # reset env if done or timeout
            next_obs = _partial_reset_and_merge_obs(env, next_obs, done_t, device)
            obs = next_obs

        # ===== bootstrap value =====
        with torch.no_grad():
            last_v = net.forward(obs)["value"]  # (N,)

        # ===== GAE / returns =====
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
            # Figure out which rows are step0 in this flattened view:
            # Original layout is (H,N) -> flattened to (H*N,)
            # Rows [0*N : 1*N) correspond to t=0
            H, N = buffer.h, buffer.n
            is_step0_mask = (idx // N) == 0  # boolean per-sample

            # Recompute logp & entropy under NEW params (mixture by timestep type)
            logp_new = torch.empty_like(old_logp_b)
            ent_new  = torch.empty_like(old_logp_b)

            if bool(is_step0_mask.any()):
                mask = is_step0_mask
                lp0, en0 = net.log_prob_recompute(obs_b[mask], act_b[mask], is_step0=True)
                logp_new[mask] = lp0
                ent_new[mask]  = en0
            if bool((~is_step0_mask).any()):
                mask = ~is_step0_mask
                lp1, en1 = net.log_prob_recompute(obs_b[mask], act_b[mask], is_step0=False)
                logp_new[mask] = lp1
                ent_new[mask]  = en1

            ratio = torch.exp(logp_new - old_logp_b)

            adv_b = adv_flat[idx]
            ret_b = ret_flat[idx]

            surrogate1 = ratio * adv_b
            surrogate2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * adv_b
            policy_loss = -torch.min(surrogate1, surrogate2).mean()

            # Clipped value loss (same as your original)
            # Re-run value head:
            v_b = net.forward(obs_b)["value"]
            value_clipped = old_v_b + (v_b - old_v_b).clamp(-clip_eps, clip_eps)
            value_losses = (v_b - ret_b).pow(2)
            value_losses_clipped = (value_clipped - ret_b).pow(2)
            value_loss = 0.5 * torch.max(value_losses, value_losses_clipped).mean()

            loss = policy_loss + value_coef * value_loss - entropy_coef * ent_new.mean()

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 0.5)
            optimizer.step()

            policy_loss_epoch += float(policy_loss.detach())
            value_loss_epoch  += float(value_loss.detach())
            entropy_epoch     += float(ent_new.mean().detach())
            num_updates       += 1

        mean_return = float(ret.mean().detach())
        print(f"[E{epoch:04d}] return={mean_return:7.3f}  "
              f"pi={policy_loss_epoch/max(1,num_updates):7.4f}  "
              f"v={value_loss_epoch/max(1,num_updates):7.4f}  "
              f"H={entropy_epoch/max(1,num_updates):7.4f}")

        # Explained variance
        ev = float(1.0 - torch.var(ret_flat - buffer.values.view(-1)) / (torch.var(ret_flat) + 1e-12))

        # TB logs
        writer.add_scalar("return_mean", mean_return, epoch)
        writer.add_scalar("policy_loss", policy_loss_epoch / max(1, num_updates), epoch)
        writer.add_scalar("value_loss", value_loss_epoch / max(1, num_updates), epoch)
        writer.add_scalar("entropy", entropy_epoch / max(1, num_updates), epoch)
        writer.add_scalar("explained_variance", ev, epoch)
        if len(finished_episode_returns) > 0:
            mean_ep_return = float(sum(finished_episode_returns) / len(finished_episode_returns))
            writer.add_scalar("episode_return_mean", mean_ep_return, epoch)
        if len(finished_episode_track_returns) > 0:
            mean_ep_track_return = float(sum(finished_episode_track_returns) / len(finished_episode_track_returns))
            writer.add_scalar("episode_tracking_return_mean", mean_ep_track_return, epoch)

        # Save best & periodic
        if mean_return > best_mean_return:
            best_mean_return = mean_return
            torch.save({
                "model": net.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
                "best_return": best_mean_return
            }, ckpt_best)

        if (save_every > 0) and (epoch % save_every == 0):
            ckpt_path = run_dir / f"epoch_{epoch:04d}_ret_{mean_return:.2f}.pt"
            torch.save({
                "model": net.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
                "return": mean_return
            }, ckpt_path)

    return {"best_mean_return": best_mean_return, "ckpt_best": ckpt_best}
