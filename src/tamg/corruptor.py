"""TAMG corruptor — manifold-preserving adversarial latent corruption.

Applies learned operator basis to produce near-miss worlds:
Z' = Z + Σ_k α_k · O_k(Z) with sparse mask M.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from .operators import OperatorBasis
from .mutations.families import (
    identity_continuity_mutation,
    locality_mutation,
    invariant_subspace_mutation,
    temporal_splice_mutation,
    CycleBreakingMutation,
)
from src.utils.base import BaseModel


class TAMGCorruptor(BaseModel):
    """Tangent-Space Adversarial Mutation Grammar corruptor.

    Produces manifold-preserving corruptions: moves along the data
    manifold while crossing the causal-validity boundary.

    Combines learned operator basis with explicit mutation families.
    """

    def __init__(
        self,
        slot_dim: int = 64,
        content_dim: int = 32,
        num_operators: int = 16,
        d_action: int = 11,
        use_operators: bool = True,
        use_mutations: bool = True,
    ):
        super().__init__()
        self.slot_dim = slot_dim
        self.content_dim = content_dim
        self.use_operators = use_operators
        self.use_mutations = use_mutations

        # Learned operator basis
        self.operator_basis = None
        if use_operators:
            self.operator_basis = OperatorBasis(
                num_operators=num_operators,
                slot_dim=slot_dim,
            )

        # Cycle-breaking mutation (requires training)
        self.cycle_breaker = CycleBreakingMutation(slot_dim)

    def set_operator_basis(self, basis: OperatorBasis) -> None:
        """Set or replace the operator basis."""
        self.operator_basis = basis
        self.use_operators = True

    def forward(
        self, Z: torch.Tensor, A: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, dict]:
        """Apply TAMG corruptions to a worldline.

        Args:
            Z: Worldline slab (B, H, N, d).
            A: Optional action embeddings (B, H, d_action).

        Returns:
            Z_corrupted: Corrupted worldline.
            info: Dict with corruption metadata.
        """
        B, H, N, d = Z.shape
        Z_mut = Z.clone()
        info: dict = {}

        # Apply multiple corruption strategies
        # Strategy 1: Learned operator basis
        if self.use_operators and self.operator_basis is not None:
            Z_op, alphas = self.operator_basis(Z, return_coefficients=True)
            Z_mut = Z_mut + 0.5 * (Z_op - Z)  # 50% operator contribution
            info["operator_alphas"] = alphas.detach()

        # Strategy 2: Explicit mutation families (stochastic selection)
        if self.use_mutations:
            strategies = torch.rand(4)  # random selection weights
            total = strategies.sum()
            if total > 0:
                strategies = strategies / total

                # Identity-continuity
                if torch.rand(1).item() < 0.3:
                    Z_id = identity_continuity_mutation(Z, self.content_dim)
                    Z_mut = Z_mut + 0.3 * (Z_id - Z)
                    info["mutation_identity"] = True

                # Locality
                if A is not None and torch.rand(1).item() < 0.3:
                    Z_loc = locality_mutation(Z, A)
                    Z_mut = Z_mut + 0.2 * (Z_loc - Z)
                    info["mutation_locality"] = True

                # Invariant subspace
                if torch.rand(1).item() < 0.3:
                    Z_inv = invariant_subspace_mutation(Z, self.content_dim)
                    Z_mut = Z_mut + 0.3 * (Z_inv - Z)
                    info["mutation_invariant"] = True

                # Temporal splice
                if torch.rand(1).item() < 0.2:
                    Z_splice = temporal_splice_mutation(Z)
                    Z_mut = Z_mut + 0.3 * (Z_splice - Z)
                    info["mutation_splice"] = True

                # Cycle-breaking
                if A is not None and torch.rand(1).item() < 0.2:
                    Z_cycle = self.cycle_breaker.apply(Z, A)
                    Z_mut = Z_mut + 0.2 * (Z_cycle - Z)
                    info["mutation_cycle"] = True

        # Clamp to reasonable range
        Z_mut = torch.clamp(Z_mut, -10.0, 10.0)

        # Compute edit distance
        info["edit_distance"] = (Z_mut - Z).pow(2).mean().item()

        return Z_mut, info

    def mutate_batch(
        self, Z: torch.Tensor, A: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Convenience: mutate and return only Z."""
        Z_mut, _ = self.forward(Z, A)
        return Z_mut
