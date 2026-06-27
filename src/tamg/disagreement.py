"""Validator disagreement score (Section 6.5).

D(τ') = Var_k[V_k(τ')] + Σ_k 1[V_k differs from V_1 by > γ_k]

Structured disagreement — local validators accept, global validators reject —
identifies the causal-validity boundary in a fully self-supervised manner.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional
from .validators.committee import ValidatorCommittee


class DisagreementScorer:
    """Computes validator disagreement as a signal for causal invalidity.

    High disagreement = likely causal violation.
    Structured disagreement (local accept + global reject) = strong signal.
    """

    def __init__(
        self,
        gamma_thresholds: Optional[List[float]] = None,
        disagreement_threshold: float = 0.3,
    ):
        # Default per-validator gamma thresholds
        self.gamma_thresholds = gamma_thresholds or [
            0.1, 0.2, 0.15, 0.1, 0.15, 0.2, 0.1, 0.5,
        ]
        self.disagreement_threshold = disagreement_threshold

    def compute_d_score(
        self, validator_scores: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Compute disagreement score.

        D(τ') = Var_k[V_k(τ')] + Σ_k 1[V_k differs from V_1 by > γ_k]

        Args:
            validator_scores: Dict mapping name → (B,) tensor.

        Returns:
            Disagreement score per batch, shape (B,).
        """
        if not validator_scores:
            return torch.zeros(1)

        # Stack all scores
        score_list = list(validator_scores.values())
        scores = torch.stack(score_list, dim=-1)  # (B, K)

        # Variance across validators
        variance = scores.var(dim=-1)  # (B,)

        # Indicator: each validator differs from V1 by more than gamma
        v1_scores = score_list[0].unsqueeze(-1)  # (B, 1)
        gamma = torch.tensor(self.gamma_thresholds[:len(score_list)],
                             device=v1_scores.device).unsqueeze(0)  # (1, K)
        differs = (scores - v1_scores).abs() > gamma  # (B, K)
        indicator_sum = differs.float().sum(dim=-1)  # (B,)

        return variance + indicator_sum

    def is_candidate_near_miss(
        self,
        scores: Dict[str, torch.Tensor],
        w_accept: torch.Tensor,
        committee: ValidatorCommittee,
        Z: torch.Tensor,
        z0: torch.Tensor,
        A: torch.Tensor,
    ) -> torch.Tensor:
        """Identify candidate near-miss worlds.

        Conditions:
          1. W_θ assigns low energy (high acceptance)
          2. Local validators (V1, V7) accept
          3. At least one global validator (V2, V3, V4, V6) rejects

        Args:
            scores: All validator scores.
            w_accept: World model acceptance scores.
            committee: Validator committee.
            Z, z0, A: Standard inputs.

        Returns:
            Boolean mask of candidate near-misses, shape (B,).
        """
        local = committee.get_local_scores(Z, z0, A)  # (B, 2)
        global_s = committee.get_global_scores(Z, z0, A)  # (B, 4)

        local_accept = local.min(dim=-1).values > 0.5
        global_reject = global_s.max(dim=-1).values < 0.5
        w_accept_high = w_accept > 0.5

        return local_accept & global_reject & w_accept_high

    def check_structured_disagreement(
        self,
        scores: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Check for structured disagreement pattern.

        Returns:
            is_structured: Boolean mask, shape (B,).
            strength: Disagreement magnitude, shape (B,).
        """
        # Extract local and global groups
        all_names = list(scores.keys())
        local_names = ["v1_local", "v7_augmentation"] if "v7_augmentation" in all_names else all_names[:2]
        global_names = [n for n in all_names if n not in local_names][:4]

        local_scores = torch.stack([scores[n] for n in local_names if n in scores], dim=-1)
        global_scores = torch.stack([scores[n] for n in global_names if n in scores], dim=-1)

        if local_scores.shape[-1] == 0 or global_scores.shape[-1] == 0:
            return torch.zeros(scores[list(scores.keys())[0]].shape[0], dtype=torch.bool), torch.zeros(1)

        # Local accept (high scores)
        local_accept = local_scores.mean(dim=-1) > 0.5
        # Global reject (low scores)
        global_reject = global_scores.mean(dim=-1) < 0.5

        is_structured = local_accept & global_reject
        strength = local_scores.mean(dim=-1) - global_scores.mean(dim=-1)

        return is_structured, strength
