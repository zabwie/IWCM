"""Counterfactual consistency constraint head — always-active version.

Uses temporal consistency comparison + per-state variance to detect
causally invalid worldlines where features change inconsistently.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from .base import ConstraintHead


class CounterfactualHead(ConstraintHead):
    name = "counterfactual"

    def __init__(self, d_state: int, d_action: int = 11, hidden_dim: int = 256):
        super().__init__(d_state, d_action, hidden_dim)
        self.state_proj = nn.Linear(d_state, hidden_dim)
        self.comparator = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Tanh(),
        )

    def forward(self, z0: torch.Tensor, A: torch.Tensor, Z: torch.Tensor) -> torch.Tensor:
        B, H, d = Z.shape
        if H < 2:
            return torch.zeros(B, device=Z.device)
        Z_proj = self.state_proj(Z)
        early = Z_proj[:, :H//2].mean(dim=1)
        late = Z_proj[:, H//2:].mean(dim=1)
        consistency = self.comparator(torch.cat([early, late], dim=-1)).squeeze(-1)
        temporal_var = Z.var(dim=1).mean(dim=-1)
        return 0.3 * consistency + 0.7 * torch.tanh(temporal_var * 0.1)
