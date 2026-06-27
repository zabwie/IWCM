"""IWCM energy function E_θ — composes all constraint heads.

E_θ(z0, A, Z) = Σ_k λ_k · C_k(z0, A, Z)

This is the core of the IWCM: a differentiable energy function over
complete latent worldlines that replaces autoregressive transition.
"""

import torch
import torch.nn as nn
from typing import Dict, Optional

from .constraints.base import ConstraintHead, ConstraintRegistry
from .constraints.boundary import BoundaryHead
from .constraints.local_transition import LocalTransitionHead
from .constraints.invariant import InvariantHead
from .constraints.effect import EffectHead
from .constraints.counterfactual import CounterfactualHead
from src.utils.base import BaseModel


class IWCMEnergy(BaseModel):
    """Full IWCM energy function with all constraint heads.

    Composes 5 constraint heads with configurable weights λ₁–λ₅.
    Provides both the total energy and per-head breakdown for logging.

    Args:
        d_state: State encoding dimension.
        d_action: Action encoding dimension.
        hidden_dim: Hidden dimension for transformer layers.
        lambdas: Weights for each constraint head.
    """

    def __init__(
        self,
        d_state: int,
        d_action: int = 11,
        hidden_dim: int = 256,
        lambdas: Optional[Dict[str, float]] = None,
    ):
        super().__init__()
        self.d_state = d_state
        self.d_action = d_action

        # Default constraint weights (from paper)
        default_lambdas = {
            "boundary": 1.0,
            "local": 1.0,
            "invariant": 1.5,
            "effect": 1.0,
            "counterfactual": 0.5,
        }
        self.lambdas = {**default_lambdas, **(lambdas or {})}

        # Constraint heads
        self.boundary_head = BoundaryHead(d_state, d_action, hidden_dim)
        self.local_head = LocalTransitionHead(d_state, d_action, hidden_dim)
        self.invariant_head = InvariantHead(d_state, d_action, hidden_dim)
        self.effect_head = EffectHead(d_state, d_action, hidden_dim)
        self.counterfactual_head = CounterfactualHead(d_state, d_action, hidden_dim)

    def forward(
        self, z0: torch.Tensor, A: torch.Tensor, Z: torch.Tensor
    ) -> torch.Tensor:
        """Compute total energy for a batch of worldlines.

        Args:
            z0: (B, d_state) encoded initial state.
            A: (B, H, d_action) action sequence.
            Z: (B, H, d_state) state sequence.

        Returns:
            Energy per batch element, shape (B,).
        """
        energy = torch.zeros(z0.shape[0], device=z0.device)

        for name, head, weight in [
            ("boundary", self.boundary_head, self.lambdas["boundary"]),
            ("local", self.local_head, self.lambdas["local"]),
            ("invariant", self.invariant_head, self.lambdas["invariant"]),
            ("effect", self.effect_head, self.lambdas["effect"]),
            ("counterfactual", self.counterfactual_head, self.lambdas["counterfactual"]),
        ]:
            score = head(z0, A, Z)
            energy = energy + weight * score

        return energy

    def compute_per_head(
        self, z0: torch.Tensor, A: torch.Tensor, Z: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Compute per-head energy breakdown.

        Args:
            z0, A, Z: As in forward().

        Returns:
            Dict mapping head name to scalar energy.
        """
        return {
            "boundary": self.lambdas["boundary"] * self.boundary_head(z0, A, Z),
            "local": self.lambdas["local"] * self.local_head(z0, A, Z),
            "invariant": self.lambdas["invariant"] * self.invariant_head(z0, A, Z),
            "effect": self.lambdas["effect"] * self.effect_head(z0, A, Z),
            "counterfactual": self.lambdas["counterfactual"] * self.counterfactual_head(z0, A, Z),
        }

    def score_acceptance(
        self, z0: torch.Tensor, A: torch.Tensor, Z: torch.Tensor
    ) -> torch.Tensor:
        """Compute acceptance score ∈ [0, 1] from energy.

        Lower energy → higher acceptance. Uses sigmoid on negated energy.

        Args:
            z0, A, Z: As in forward().

        Returns:
            Acceptance probability per batch, shape (B,).
        """
        energy = self.forward(z0, A, Z)
        return torch.sigmoid(-energy)  # lower energy = higher acceptance
