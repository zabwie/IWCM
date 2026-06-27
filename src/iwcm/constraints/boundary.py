"""Boundary constraint head — anchors all future states to real z0."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .base import ConstraintHead


class BoundaryHead(ConstraintHead):
    def __init__(self, d_state: int, d_action: int = 11, hidden_dim: int = 256, num_layers: int = 2):
        super().__init__(d_state, d_action, hidden_dim)
        self.z0_proj = nn.Linear(d_state, hidden_dim)
        self.state_proj = nn.Linear(d_state, hidden_dim)
        self.scorer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, z0: torch.Tensor, A: torch.Tensor, Z: torch.Tensor) -> torch.Tensor:
        B, H, d = Z.shape
        z0_feat = self.z0_proj(z0)  # (B, hidden)
        Z_feat = self.state_proj(Z)  # (B, H, hidden)

        # Cross-attention via dot product: z0 attends to each z_t
        attn_weights = torch.einsum("bd,bhd->bh", z0_feat, Z_feat)  # (B, H)
        attn_weights = F.softmax(attn_weights / (self.hidden_dim ** 0.5), dim=-1)

        # Weighted combination of Z features
        z_ctx = (attn_weights.unsqueeze(-1) * Z_feat).sum(dim=1)  # (B, hidden)

        # Score: how well does z0 predict the context?
        score = self.scorer(z_ctx).squeeze(-1)  # (B,)

        # Also penalize mean drift from z0
        z0_exp = z0.unsqueeze(1).expand(-1, H, -1)
        drift = (Z - z0_exp).pow(2).mean(dim=(-2, -1))
        return score + 0.1 * drift
