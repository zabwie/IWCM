"""Counterfactual consistency constraint head.

C_counterfactual: objects must maintain identity-invariant features
across the worldline. Perturbs Z slightly and checks if inconsistent
changes are detected — penalizing worldlines where small perturbations
cause large energy changes (i.e., the worldline is on a causal knife-edge).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .base import ConstraintHead


class CounterfactualHead(ConstraintHead):
    def __init__(self, d_state: int, d_action: int = 11, hidden_dim: int = 256):
        super().__init__(d_state, d_action, hidden_dim)
        self.state_proj = nn.Linear(d_state, hidden_dim)
        self.scorer = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, z0: torch.Tensor, A: torch.Tensor, Z: torch.Tensor) -> torch.Tensor:
        B, H, d = Z.shape
        if H < 2:
            return torch.zeros(B, device=Z.device)

        # Temporal consistency: compare early vs late worldline segments
        Z_proj = self.state_proj(Z)  # (B, H, hidden)
        early = Z_proj[:, :H//2].mean(dim=1)  # (B, hidden)
        late = Z_proj[:, H//2:].mean(dim=1)   # (B, hidden)
        pair = torch.cat([early, late], dim=-1)  # (B, 2*hidden)

        # Score: how inconsistent are early and late? High score = violation
        score = self.scorer(pair).squeeze(-1)  # (B,)

        # Also check: does adding noise to Z cause disproportionate energy changes?
        noise = torch.randn(B, H, d, device=Z.device) * 0.01
        Z_noisy = Z + noise
        noisy_proj = self.state_proj(Z_noisy).mean(dim=1)
        orig_proj = self.state_proj(Z).mean(dim=1)
        noise_sensitivity = (noisy_proj - orig_proj).norm(dim=-1)  # (B,)

        return score + 0.1 * noise_sensitivity
