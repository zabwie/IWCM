"""AC3 hardness scoring and curriculum management.

H(τ') = W_θ^accept(τ') + d_surface(τ, τ') - d_edit(τ, τ') + O^violation(τ')

The curriculum targets the model's uncertainty boundary [0.4, 0.7]
for maximum learning signal.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Tuple, Optional


class HardnessScorer:
    """Computes hardness score for corrupted trajectories.

    The hardness score measures how "useful" a corrupted trajectory is
    for training — hard enough to challenge the model, but not so hard
    that it's irrelevant.
    """

    def __init__(self, weight_accept: float = 1.0, weight_surface: float = 0.5,
                 weight_edit: float = -0.3, weight_violation: float = 1.0):
        self.weight_accept = weight_accept
        self.weight_surface = weight_surface
        self.weight_edit = weight_edit
        self.weight_violation = weight_violation

    def compute(
        self,
        accept_scores: torch.Tensor,       # (B,)
        surface_distances: torch.Tensor,   # (B,)
        edit_distances: torch.Tensor,      # (B,)
        violation_counts: torch.Tensor,    # (B,) — from oracle
    ) -> torch.Tensor:
        """Compute hardness scores for a batch.

        Args:
            accept_scores: W_θ acceptance ∈ [0, 1].
            surface_distances: Distance between original and corrupted.
            edit_distances: Magnitude of edits applied.
            violation_counts: Number of oracle-detected violations.

        Returns:
            Hardness scores, shape (B,).
        """
        return (
            self.weight_accept * accept_scores +
            self.weight_surface * surface_distances -
            self.weight_edit * edit_distances +
            self.weight_violation * violation_counts
        )

    def select_hard_negatives(
        self,
        corruptions: List,
        accept_scores: torch.Tensor,
        surface_distances: torch.Tensor,
        edit_distances: torch.Tensor,
        violation_counts: torch.Tensor,
        top_k: int = 8,
        accept_low: float = 0.4,
        accept_high: float = 0.7,
    ) -> Tuple[List, torch.Tensor]:
        """Select top-k hardest negatives within acceptance range.

        Only corruptions with acceptance ∈ [accept_low, accept_high]
        are considered — these are at the model's uncertainty boundary.

        Returns:
            hard_corruptions: List of selected corrupted items.
            indices: Indices of selected items.
        """
        hardness = self.compute(accept_scores, surface_distances,
                                edit_distances, violation_counts)

        # Filter to acceptance range
        in_range = (accept_scores >= accept_low) & (accept_scores <= accept_high)

        if in_range.sum() == 0:
            # Fall back to all if none in range
            in_range = torch.ones_like(accept_scores, dtype=torch.bool)

        hardness[~in_range] = -float("inf")

        k = min(top_k, len(corruptions))
        _, indices = hardness.topk(k)

        hard = [corruptions[i] for i in indices.cpu().tolist()]
        return hard, indices


class CurriculumManager:
    """Manages the adversarial curriculum across training.

    Tracks model acceptance rate and adjusts corruption difficulty
    to maintain the optimal learning zone.
    """

    def __init__(
        self,
        accept_low: float = 0.4,
        accept_high: float = 0.7,
        top_k: int = 8,
        warmup_steps: int = 1000,
    ):
        self.accept_low = accept_low
        self.accept_high = accept_high
        self.top_k = top_k
        self.warmup_steps = warmup_steps

        # Tracking
        self.step = 0
        self.acceptance_history: List[float] = []

    def should_apply_curriculum(self) -> bool:
        """Check if curriculum is active (past warmup)."""
        return self.step >= self.warmup_steps

    def update(self, accept_scores: torch.Tensor) -> None:
        """Update curriculum with latest batch acceptance."""
        self.step += 1
        self.acceptance_history.append(accept_scores.mean().item())

        # Keep only recent history
        if len(self.acceptance_history) > 1000:
            self.acceptance_history = self.acceptance_history[-1000:]

    def get_mean_acceptance(self) -> float:
        """Current mean acceptance rate."""
        if not self.acceptance_history:
            return 0.0
        return float(np.mean(self.acceptance_history[-100:]))

    def is_in_range(self, accept_scores: torch.Tensor) -> torch.Tensor:
        """Check which samples are in the optimal acceptance range."""
        return (accept_scores >= self.accept_low) & (accept_scores <= self.accept_high)

    def step(self, batch_size: int) -> int:
        """Increment step counter. Returns current step."""
        self.step += batch_size
        return self.step
