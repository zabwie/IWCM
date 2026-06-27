"""IWCM learned parallel refinement operator.

Z^{(k+1)} = Φ_θ(z0, A, Z^{(k)}, r^{(k)})

A learned transformer module that refines the worldline in parallel,
using constraint residuals r^{(k)} as conditioning. More efficient than
pure gradient descent for fixed-step refinement.
"""

import torch
import torch.nn as nn
from typing import Tuple

from .energy import IWCMEnergy
from src.utils.base import BaseModel


class LearnedRefinementOperator(BaseModel):
    """Learned parallel refinement operator for IWCM.

    Takes current worldline Z and constraint residuals, produces an
    improved worldline in a single forward pass (learned step).

    Can be used standalone or in combination with gradient descent.
    """

    def __init__(
        self,
        d_state: int,
        d_action: int = 11,
        hidden_dim: int = 256,
        num_layers: int = 2,
        num_heads: int = 4,
    ):
        super().__init__()
        self.d_state = d_state
        self.d_action = d_action
        self.hidden_dim = hidden_dim

        # Encode (z0, Z, A, residuals) into hidden
        self.z0_proj = nn.Linear(d_state, hidden_dim)
        self.state_proj = nn.Linear(d_state, hidden_dim)
        self.action_proj = nn.Linear(d_action, hidden_dim)
        self.residual_proj = nn.Linear(5, hidden_dim)  # 5 constraint residuals

        # Transformer over worldline
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Output: delta to apply to Z
        self.output_proj = nn.Linear(hidden_dim, d_state)

    def compute_residuals(
        self, energy_fn: IWCMEnergy,
        z0: torch.Tensor, A: torch.Tensor, Z: torch.Tensor,
    ) -> torch.Tensor:
        """Compute per-head constraint residuals.

        Returns:
            Residuals tensor of shape (B, H, 5).
        """
        with torch.no_grad():
            per_head = energy_fn.compute_per_head(z0, A, Z)
        # Stack residuals: (B, H, 5)
        residuals = torch.stack([
            per_head["boundary"].unsqueeze(1).expand(-1, Z.shape[1]),
            per_head["local"].unsqueeze(1).expand(-1, Z.shape[1]),
            per_head["invariant"].unsqueeze(1).expand(-1, Z.shape[1]),
            per_head["effect"].unsqueeze(1).expand(-1, Z.shape[1]),
            per_head["counterfactual"].unsqueeze(1).expand(-1, Z.shape[1]),
        ], dim=-1)
        return residuals

    def forward(
        self, z0: torch.Tensor, A: torch.Tensor, Z: torch.Tensor, residuals: torch.Tensor
    ) -> torch.Tensor:
        """Apply learned refinement to Z.

        Args:
            z0: (B, d_state).
            A: (B, H, d_action).
            Z: (B, H, d_state) — current worldline.
            residuals: (B, H, 5) — constraint residuals.

        Returns:
            Refined Z', shape (B, H, d_state).
        """
        B, H, _ = Z.shape

        # Encode all inputs
        z0_enc = self.z0_proj(z0).unsqueeze(1)  # (B, 1, hidden)
        Z_enc = self.state_proj(Z)  # (B, H, hidden)
        A_enc = self.action_proj(A)  # (B, H, hidden)
        r_enc = self.residual_proj(residuals)  # (B, H, hidden)

        # Combine: prepend z0 token
        tokens = torch.cat([z0_enc, Z_enc + A_enc + r_enc], dim=1)  # (B, H+1, hidden)

        # Transformer refinement
        refined = self.transformer(tokens)  # (B, H+1, hidden)

        # Extract Z refinement (skip z0 token)
        delta = self.output_proj(refined[:, 1:, :])  # (B, H, d_state)

        return Z + delta


class RefinementSolver(nn.Module):
    """Hybrid solver combining gradient descent and learned refinement.

    Uses gradient descent for the first half of steps (exploration),
    then learned refinement for the final steps (exploitation).
    """

    def __init__(
        self,
        energy_fn: IWCMEnergy,
        gd_steps: int = 10,
        gd_lr: float = 0.01,
        ref_op: LearnedRefinementOperator = None,
        ref_steps: int = 3,
    ):
        super().__init__()
        self.energy_fn = energy_fn
        self.gd_steps = gd_steps
        self.gd_lr = gd_lr
        self.ref_op = ref_op
        self.ref_steps = ref_steps

    def solve(self, z0: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        """Refine worldline using hybrid approach.

        Args:
            z0: (B, d_state).
            A: (B, H, d_action).

        Returns:
            Refined Z, shape (B, H, d_state).
        """
        B = z0.shape[0]
        H = A.shape[1]
        d = self.energy_fn.d_state

        # Phase 1: Gradient descent
        Z = torch.randn(B, H, d, device=z0.device, requires_grad=True)
        velocity = torch.zeros_like(Z)
        for _ in range(self.gd_steps):
            energy = self.energy_fn(z0, A, Z).mean()
            grad = torch.autograd.grad(energy, Z, create_graph=False)[0]
            velocity = 0.9 * velocity + grad
            Z = Z.detach() - self.gd_lr * velocity
            Z.requires_grad_(True)
            velocity = velocity.detach()

        # Phase 2: Learned refinement
        if self.ref_op is not None:
            for _ in range(self.ref_steps):
                residuals = self.ref_op.compute_residuals(self.energy_fn, z0, A, Z)
                Z = self.ref_op(z0, A, Z, residuals)

        return Z

    def forward(self, z0: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        return self.solve(z0, A)
