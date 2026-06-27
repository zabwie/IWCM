"""Action-effect locality constraint head.

Implements C_effect(Z, A) = Σ_t Σ_{o ∉ scope(a_t)} Δ(z_t^o, z_{t+1}^o).

Actions may only causally affect objects within their reachable scope.
Objects outside the action's scope must not change. This constraint
discovers causal locality from data.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .base import ConstraintHead


class EffectHead(ConstraintHead):
    """Action-effect locality: actions only affect objects in their causal scope.

    Learns a causal scope per action type, then penalizes changes to
    objects outside that scope.
    """

    def __init__(
        self,
        d_state: int,
        d_action: int = 11,
        hidden_dim: int = 256,
        num_objects: int = 6,
    ):
        super().__init__(d_state, d_action, hidden_dim)

        # Learnable causal scope per action
        # scope: for each of num_objects slots, how much action k affects it
        self.scope = nn.Parameter(torch.randn(d_action, num_objects) * 0.1)

        # State change projector
        self.change_proj = nn.Sequential(
            nn.Linear(d_state * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_objects),
            nn.Sigmoid(),  # per-object change magnitude
        )

    def forward(
        self, z0: torch.Tensor, A: torch.Tensor, Z: torch.Tensor
    ) -> torch.Tensor:
        B, H, d = Z.shape

        if H < 2:
            return torch.zeros(B, device=Z.device)

        # Compute per-step state changes
        z_t = Z[:, :-1, :]  # (B, H-1, d)
        z_tp1 = Z[:, 1:, :]  # (B, H-1, d)

        # Concatenate consecutive states
        pair = torch.cat([z_t, z_tp1], dim=-1)  # (B, H-1, 2*d)
        changes = self.change_proj(pair)  # (B, H-1, num_objects)

        # Action scope: which objects each action should affect
        # A shape: (B, H, d_action) — take actions for t=0..H-2
        a_t = A[:, :H-1, :]  # (B, H-1, d_action)

        # Compute expected change per action via scope matrix
        # a_t: (B, H-1, d_action) @ scope: (d_action, num_objects)
        expected_affect = a_t @ self.scope  # (B, H-1, num_objects)
        expected_affect = torch.sigmoid(expected_affect)

        # Penalize changes where object NOT in scope
        # If action doesn't affect object, change should be 0
        out_of_scope_penalty = changes * (1.0 - expected_affect)  # (B, H-1, num_objects)
        violation = out_of_scope_penalty.mean(dim=(-2, -1))  # (B,)

        return violation
