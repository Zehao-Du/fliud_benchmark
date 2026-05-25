"""HDF5 demo dataset loader with normalization for diffusion policy training."""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

try:
    import h5py
except ImportError as e:
    raise ImportError("h5py required: pip install h5py") from e


class DemoDataset(Dataset):
    """Sliding-window demo dataset from a collect_demo_data.py HDF5 file.

    Each item contains:
        obs_seq  (To, obs_dim)  — observation history (current + To-1 past frames)
        actions  (Ta, action_dim) — future action sequence to predict
        timestep int             — step index within episode (for diagnostics)
    """

    def __init__(
        self,
        hdf5_path: str | Path,
        obs_horizon: int = 2,
        pred_horizon: int = 8,
        normalize: bool = True,
    ) -> None:
        self.obs_horizon = obs_horizon
        self.pred_horizon = pred_horizon
        self.normalize = normalize

        with h5py.File(str(hdf5_path), "r") as f:
            meta = f["metadata"]
            self.obs_dim: int = int(meta.attrs["obs_dim"])
            self.action_dim: int = int(meta.attrs["action_dim"])

            if normalize:
                self._obs_min = torch.tensor(meta["obs_min"][:], dtype=torch.float32)
                self._obs_max = torch.tensor(meta["obs_max"][:], dtype=torch.float32)
                self._act_min = torch.tensor(meta["act_min"][:], dtype=torch.float32)
                self._act_max = torch.tensor(meta["act_max"][:], dtype=torch.float32)

            # Load all episodes
            self._obs_seqs: list[torch.Tensor] = []
            self._act_seqs: list[torch.Tensor] = []

            num_episodes = int(meta.attrs["num_episodes"])
            for i in range(num_episodes):
                key = f"episode_{i:04d}"
                if key not in f:
                    continue
                grp = f[key]
                obs = torch.tensor(grp["observations/obs_vector"][:], dtype=torch.float32)
                act = torch.tensor(grp["actions/action_vector"][:], dtype=torch.float32)
                if len(obs) < obs_horizon + pred_horizon:
                    continue
                self._obs_seqs.append(obs)
                self._act_seqs.append(act)

        # Build flat index: list of (episode_idx, start_step)
        self._index: list[tuple[int, int]] = []
        for ep_idx, obs in enumerate(self._obs_seqs):
            T = len(obs)
            for t in range(T - obs_horizon - pred_horizon + 1):
                self._index.append((ep_idx, t))

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        ep_idx, t = self._index[idx]
        obs_seq = self._obs_seqs[ep_idx][t : t + self.obs_horizon]        # (To, D_obs)
        act_seq = self._act_seqs[ep_idx][t + self.obs_horizon - 1 : t + self.obs_horizon - 1 + self.pred_horizon]  # (Ta, D_act)

        if self.normalize:
            obs_seq = self._norm_obs(obs_seq)
            act_seq = self._norm_act(act_seq)

        return {
            "obs_seq": obs_seq,
            "actions": act_seq,
            "timestep": torch.tensor(t, dtype=torch.long),
        }

    # ------------------------------------------------------------------
    def _norm_obs(self, x: torch.Tensor) -> torch.Tensor:
        rng = (self._obs_max - self._obs_min).clamp(min=1e-6)
        return 2.0 * (x - self._obs_min) / rng - 1.0  # → [-1, 1]

    def _norm_act(self, x: torch.Tensor) -> torch.Tensor:
        rng = (self._act_max - self._act_min).clamp(min=1e-6)
        return 2.0 * (x - self._act_min) / rng - 1.0

    def denorm_act(self, x: torch.Tensor) -> torch.Tensor:
        """Inverse normalize actions (for inference output)."""
        rng = (self._act_max - self._act_min).clamp(min=1e-6)
        return ((x + 1.0) / 2.0) * rng + self._act_min

    @property
    def norm_stats(self) -> dict[str, torch.Tensor]:
        return {
            "obs_min": self._obs_min,
            "obs_max": self._obs_max,
            "act_min": self._act_min,
            "act_max": self._act_max,
        }
