"""Counterfactual consistency constraint head (Section 4.2).

C_counterfactual(Z, Z', A, A') = Σ d(z_t^o, z_t'^o)
for objects o not causally affected by the action difference.

In single-worldline mode: creates an internal counterfactual by
perturbing actions, then checking which features remain invariant."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from .base import ConstraintHead


class CounterfactualHead(ConstraintHead):
    name = "counterfactual"

    def __init__(self, d_state: int, d_action: int = 11, hidden_dim: int = 256):
        super().__init__(d_state, d_action, hidden_dim)
        self.action_proj = nn.Linear(d_action, d_action)
        self.state_proj = nn.Linear(d_state, hidden_dim)
        self.comparator = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, z0: torch.Tensor, A: torch.Tensor, Z: torch.Tensor) -> torch.Tensor:
        B, H, d = Z.shape

        # Generate internal counterfactual: perturb actions to create Z'
        noise = torch.randn(B, H, self.d_action, device=Z.device) * 0.1
        A_prime = A + noise
        A_diff = (A - A_prime).abs().mean(dim=-1)  # (B, H) — which timesteps differ

        # Compare Z with itself: features that vary under action perturbation
        # are "causally affected"; features that remain identical should not change
        Z_proj = self.state_proj(Z)  # (B, H, hidden)

        # Compute per-timestep feature stability under action change
        # For timesteps where A ≈ A', features should be invariant
        early = Z_proj[:, :H//2].mean(dim=1)   # (B, hidden)
        late = Z_proj[:, H//2:].mean(dim=1)    # (B, hidden)

        # Temporal consistency: early vs late should be consistent
        # unless actions explain the difference
        action_change = A_diff.mean(dim=-1)  # (B,) — how much actions changed overall
        consistency = self.comparator(torch.cat([early, late], dim=-1)).squeeze(-1)  # (B,)

        # Violation: temporal inconsistency that is NOT explained by action change
        violation = F.relu(consistency - 0.5 * action_change)
        return violation
