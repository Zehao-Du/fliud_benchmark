"""1D conditional UNet + DDPM diffusion policy.

Architecture:
  ObsEncoder:   MLP  (To * obs_dim) → embed_dim
  NoiseUNet1D:  conditional 1D UNet  (Ta * action_dim + noise) → denoised action
  DDPMScheduler: cosine noise schedule, T=100 steps

Training usage:
  policy = DiffusionPolicy(obs_dim, action_dim, To, Ta)
  loss = policy.compute_loss(obs_seq, gt_actions)   # MSE on predicted noise

Inference usage:
  pred_actions = policy.predict_action(obs_seq)      # DDPM reverse process
"""
from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Sinusoidal time embedding
# ---------------------------------------------------------------------------

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freq = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / (half - 1))
        emb = t.float().unsqueeze(1) * freq.unsqueeze(0)
        return torch.cat([emb.sin(), emb.cos()], dim=-1)


# ---------------------------------------------------------------------------
# 1D ResNet block (conditioned on time + obs embeddings)
# ---------------------------------------------------------------------------

class ResBlock1D(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, cond_dim: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(in_ch, out_ch, 3, padding=1)
        self.conv2 = nn.Conv1d(out_ch, out_ch, 3, padding=1)
        self.norm1 = nn.GroupNorm(min(8, out_ch), out_ch)
        self.norm2 = nn.GroupNorm(min(8, out_ch), out_ch)
        self.cond_proj = nn.Linear(cond_dim, out_ch * 2)
        self.res = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T)  cond: (B, cond_dim)
        h = F.silu(self.norm1(self.conv1(x)))
        scale, shift = self.cond_proj(cond).chunk(2, dim=-1)
        h = h * (1 + scale.unsqueeze(-1)) + shift.unsqueeze(-1)
        h = F.silu(self.norm2(self.conv2(h)))
        return h + self.res(x)


# ---------------------------------------------------------------------------
# 1D UNet
# ---------------------------------------------------------------------------

class NoiseUNet1D(nn.Module):
    def __init__(self, action_dim: int, cond_dim: int, base_ch: int = 64) -> None:
        super().__init__()
        chs = [base_ch, base_ch * 2, base_ch * 4]

        self.in_proj = nn.Conv1d(action_dim, chs[0], 1)

        self.down = nn.ModuleList([
            ResBlock1D(chs[0], chs[0], cond_dim),
            ResBlock1D(chs[0], chs[1], cond_dim),
            ResBlock1D(chs[1], chs[2], cond_dim),
        ])
        self.mid = ResBlock1D(chs[2], chs[2], cond_dim)
        self.up = nn.ModuleList([
            ResBlock1D(chs[2] + chs[2], chs[1], cond_dim),
            ResBlock1D(chs[1] + chs[1], chs[0], cond_dim),
            ResBlock1D(chs[0] + chs[0], chs[0], cond_dim),
        ])
        self.out_proj = nn.Conv1d(chs[0], action_dim, 1)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        # x: (B, Ta, action_dim) → rearrange to (B, action_dim, Ta)
        x = x.permute(0, 2, 1)
        x = self.in_proj(x)
        skips = []
        for blk in self.down:
            x = blk(x, cond)
            skips.append(x)
        x = self.mid(x, cond)
        for blk, skip in zip(self.up, reversed(skips)):
            x = blk(torch.cat([x, skip], dim=1), cond)
        x = self.out_proj(x)
        return x.permute(0, 2, 1)  # (B, Ta, action_dim)


# ---------------------------------------------------------------------------
# Observation encoder
# ---------------------------------------------------------------------------

class ObsEncoder(nn.Module):
    def __init__(self, obs_dim: int, obs_horizon: int, embed_dim: int) -> None:
        super().__init__()
        in_dim = obs_dim * obs_horizon
        self.net = nn.Sequential(
            nn.Linear(in_dim, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim),
            nn.SiLU(),
        )

    def forward(self, obs_seq: torch.Tensor) -> torch.Tensor:
        # obs_seq: (B, To, obs_dim)
        B = obs_seq.shape[0]
        return self.net(obs_seq.reshape(B, -1))


# ---------------------------------------------------------------------------
# DDPM schedule
# ---------------------------------------------------------------------------

class DDPMScheduler:
    def __init__(self, num_steps: int = 100, beta_schedule: str = "cosine") -> None:
        self.T = num_steps
        t = torch.linspace(0, num_steps, num_steps + 1) / num_steps
        alpha_bar = torch.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2
        alpha_bar = alpha_bar / alpha_bar[0]
        betas = (1 - alpha_bar[1:] / alpha_bar[:-1]).clamp(0, 0.999)
        alphas = 1 - betas
        alpha_cumprod = torch.cumprod(alphas, dim=0)

        self.register = lambda name, val: setattr(self, name, val)
        self.betas = betas
        self.alphas = alphas
        self.alpha_cumprod = alpha_cumprod
        self.sqrt_acp = alpha_cumprod.sqrt()
        self.sqrt_one_minus_acp = (1 - alpha_cumprod).sqrt()

    def add_noise(self, x0: torch.Tensor, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        noise = torch.randn_like(x0)
        sqrt_acp = self.sqrt_acp[t.cpu()].to(x0.device)
        sqrt_1m_acp = self.sqrt_one_minus_acp[t.cpu()].to(x0.device)
        while sqrt_acp.dim() < x0.dim():
            sqrt_acp = sqrt_acp.unsqueeze(-1)
            sqrt_1m_acp = sqrt_1m_acp.unsqueeze(-1)
        return sqrt_acp * x0 + sqrt_1m_acp * noise, noise

    def step(self, x_t: torch.Tensor, t: int, pred_noise: torch.Tensor) -> torch.Tensor:
        alpha = self.alphas[t].to(x_t.device)
        beta = self.betas[t].to(x_t.device)
        acp = self.alpha_cumprod[t].to(x_t.device)
        pred_x0 = (x_t - (1 - acp).sqrt() * pred_noise) / acp.sqrt()
        pred_x0 = pred_x0.clamp(-1.5, 1.5)
        mean = (alpha.sqrt() * (1 - (acp / alpha)) * x_t + (acp / alpha).sqrt() * beta * pred_x0) / (1 - acp)
        if t > 0:
            noise = torch.randn_like(x_t)
            prev_acp = self.alpha_cumprod[t - 1].to(x_t.device)
            variance = beta * (1 - prev_acp) / (1 - acp)
            mean = mean + variance.sqrt() * noise
        return mean


# ---------------------------------------------------------------------------
# Unified DiffusionPolicy
# ---------------------------------------------------------------------------

class DiffusionPolicy(nn.Module):
    """DDPM diffusion policy over EE-position + gripper actions.

    Args:
        obs_dim: Dimension of a single observation vector.
        action_dim: Dimension of a single action vector (default 4: 3 ee_pos + 1 gripper).
        obs_horizon: Number of past observations to condition on (To).
        pred_horizon: Number of future actions to predict (Ta).
        embed_dim: Hidden size for obs encoder and time embedding.
        unet_base_ch: Base channel count for the 1D UNet.
        num_diffusion_steps: DDPM denoising steps (T).
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int = 4,
        obs_horizon: int = 2,
        pred_horizon: int = 8,
        embed_dim: int = 256,
        unet_base_ch: int = 64,
        num_diffusion_steps: int = 100,
    ) -> None:
        super().__init__()
        self.obs_horizon = obs_horizon
        self.pred_horizon = pred_horizon
        self.action_dim = action_dim
        self.num_diffusion_steps = num_diffusion_steps

        self.obs_encoder = ObsEncoder(obs_dim, obs_horizon, embed_dim)
        self.time_emb = SinusoidalPosEmb(embed_dim)
        self.time_proj = nn.Sequential(nn.Linear(embed_dim, embed_dim), nn.SiLU())
        cond_dim = embed_dim * 2  # obs + time
        self.unet = NoiseUNet1D(action_dim, cond_dim, unet_base_ch)
        self.scheduler = DDPMScheduler(num_diffusion_steps)

    def _cond(self, obs_seq: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        obs_emb = self.obs_encoder(obs_seq)
        t_emb = self.time_proj(self.time_emb(t))
        return torch.cat([obs_emb, t_emb], dim=-1)

    def compute_loss(self, obs_seq: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """Training loss: MSE on predicted noise."""
        B = obs_seq.shape[0]
        t = torch.randint(0, self.num_diffusion_steps, (B,), device=obs_seq.device)
        x_noisy, noise = self.scheduler.add_noise(actions, t)
        cond = self._cond(obs_seq, t)
        pred_noise = self.unet(x_noisy, cond)
        return F.mse_loss(pred_noise, noise)

    @torch.no_grad()
    def predict_action(self, obs_seq: torch.Tensor, num_steps: int | None = None) -> torch.Tensor:
        """DDPM reverse process: sample action sequence from noise."""
        B = obs_seq.shape[0]
        T = self.num_diffusion_steps
        steps = num_steps or T
        # Uniformly subsample timesteps when steps < T (DDPM subsampling)
        indices = torch.linspace(T - 1, 0, steps, dtype=torch.long)
        x = torch.randn(B, self.pred_horizon, self.action_dim, device=obs_seq.device)
        for step in indices.tolist():
            step = int(step)
            t = torch.full((B,), step, device=obs_seq.device, dtype=torch.long)
            cond = self._cond(obs_seq, t)
            pred_noise = self.unet(x, cond)
            x = self.scheduler.step(x, step, pred_noise)
        return x
