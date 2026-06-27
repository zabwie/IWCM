"""Counterfactual consistency constraint head.

Implements C_counterfactual(Z, Z', A, A') — for two futures from the same z0,
objects not causally reachable by the action difference must remain identical.

This is the most sophisticated constraint head and is critical for
causal law discovery.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .base import ConstraintHead


class CounterfactualHead(ConstraintHead):
    """Counterfactual consistency: unaffected objects must stay identical.

    Takes TWO worldlines (Z, Z') from the same z0 with different actions
    (A, A'), and penalizes changes to objects not in the action difference.
    """

    def __init__(
        self,
        d_state: int,
        d_action: int = 11,
        hidden_dim: int = 256,
    ):
        super().__init__(d_state, d_action, hidden_dim)

        # Action difference encoder
        self.action_diff_proj = nn.Linear(d_action, hidden_dim)

        # State comparison: which objects changed between Z and Z'
        self.comparison = nn.Sequential(
            nn.Linear(d_state * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward_standard(
        self, z0: torch.Tensor, A: torch.Tensor, Z: torch.Tensor
    ) -> torch.Tensor:
        """Standard forward (single worldline) — returns zero by default.

        Counterfactual head only produces meaningful output with paired inputs.
        """
        return torch.zeros(z0.shape[0], device=z0.device)

    def forward_paired(
        self,
        z0: torch.Tensor,
        A: torch.Tensor,
        Z: torch.Tensor,
        A2: torch.Tensor,
        Z2: torch.Tensor,
    ) -> torch.Tensor:
        """Paired forward — compare two futures from the same z0.

        Args:
            z0: Initial state, shape (B, d_state).
            A: Action sequence for worldline 1, shape (B, H, d_action).
            Z: Worldline 1, shape (B, H, d_state).
            A2: Action sequence for worldline 2, shape (B, H, d_action).
            Z2: Worldline 2, shape (B, H, d_state).

        Returns:
            Violation score per batch, shape (B,).
        """
        B, H, d = Z.shape

        # Compute action difference
        A_diff = (A - A2).mean(dim=1)  # (B, d_action)

        # Compare states between worldlines
        # Concatenate corresponding states from both worldlines
        pair_features = torch.cat([Z, Z2], dim=-1)  # (B, H, 2*d)
        per_step_diff = self.comparison(pair_features).squeeze(-1)  # (B, H)

        # Mean difference over time
        state_diff = per_step_diff.mean(dim=-1)  # (B,)

        # The violation: state differences that are NOT explained by action differences
        # If actions differ, state changes are expected. If actions are similar,
        # state changes indicate inconsistency.
        action_magnitude = A_diff.abs().mean(dim=-1, keepdim=False)  # (B,)

        # Violation = state_diff * (1 - action_magnitude)
        # i.e., penalize state changes when actions are similar
        violation = state_diff * (1.0 - torch.tanh(action_magnitude))  # (B,)

        return violation
