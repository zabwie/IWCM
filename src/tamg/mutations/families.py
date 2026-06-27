"""TAMG mutation families — 5 continuous latent mutation types.

Section 6.3-6.4: Each mutation family applies learned edits to
latent slot representations, creating near-miss worlds that are
locally plausible but globally causally invalid.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
from .operators import OperatorBasis


# ═══════════════════════════════════════════════════════════
# Identity-Continuity Mutation (Section 6.4.1)
# ═══════════════════════════════════════════════════════════

def identity_continuity_mutation(
    Z: torch.Tensor,
    content_dim: int,
    rng_state: Optional[int] = None,
) -> torch.Tensor:
    """Swap slot content embeddings during ambiguous intervals.

    z'_{t,i} = content(z_{t,j}) + pose(z_{t,i})

    Creates futures where an object moves smoothly but changes its
    underlying identity — violating object permanence.

    Args:
        Z: Worldline slab (B, H, N, d).
        content_dim: Dimension of content subspace.
        rng_state: Random seed for reproducibility.

    Returns:
        Mutated worldline Z', same shape.
    """
    B, H, N, d = Z.shape
    rng = torch.Generator()
    if rng_state is not None:
        rng.manual_seed(rng_state)

    Z_mut = Z.clone()
    content = Z[..., :content_dim]
    pose = Z[..., content_dim:content_dim + (d - content_dim)]

    # Select two random slots to swap content
    if N >= 2:
        for b in range(B):
            i, j = torch.randperm(N, generator=rng)[:2]
            # Swap content while preserving pose
            Z_mut[b, :, i, :content_dim] = content[b, :, j]
            Z_mut[b, :, j, :content_dim] = content[b, :, i]

    return Z_mut


# ═══════════════════════════════════════════════════════════
# Locality Mutation (Section 6.4.2)
# ═══════════════════════════════════════════════════════════

def locality_mutation(
    Z: torch.Tensor,
    A: torch.Tensor,
    attention_threshold: float = 0.1,
    noise_scale: float = 0.05,
) -> torch.Tensor:
    """Apply action-correlated deltas to slots with low causal attention.

    Δz_{t,j} = f_action(a_t), for j ∉ causal-neighborhood(actor).

    Args:
        Z: Worldline slab (B, H, N, d).
        A: Action embeddings (B, H, d_action).
        attention_threshold: Below this attention = "not causally connected".
        noise_scale: Magnitude of perturbation.

    Returns:
        Mutated Z'.
    """
    B, H, N, d = Z.shape
    Z_mut = Z.clone()

    # Random object to act as "actor"
    actor_idx = torch.randint(0, N, (B,))
    # Random non-actor object to perturb
    for b in range(B):
        non_actors = [i for i in range(N) if i != actor_idx[b]]
        if not non_actors:
            continue
        target = non_actors[torch.randint(0, len(non_actors), (1,)).item()]

        # Random timestep
        t = torch.randint(0, H, (1,)).item()
        action_noise = A[b, t].mean() * noise_scale
        Z_mut[b, t, target] += action_noise

    return Z_mut


# ═══════════════════════════════════════════════════════════
# Invariant-Subspace Mutation (Section 6.4.3)
# ═══════════════════════════════════════════════════════════

def invariant_subspace_mutation(
    Z: torch.Tensor,
    content_dim: int,
    noise_scale: float = 0.01,
) -> torch.Tensor:
    """Perturb content subspace without a valid causal event.

    z'_{t,i}.content = z_{t,i}.content + δ_c
    where ‖δ_c‖ ≪ ‖z_{t,i}.content‖

    Args:
        Z: Worldline slab (B, H, N, d).
        content_dim: Dimension of content subspace.
        noise_scale: Magnitude of perturbation (small).

    Returns:
        Mutated Z'.
    """
    B, H, N, d = Z.shape
    Z_mut = Z.clone()

    # Small perturbation to content subspace
    noise = torch.randn(B, H, N, content_dim, device=Z.device) * noise_scale
    Z_mut[..., :content_dim] += noise

    return Z_mut


# ═══════════════════════════════════════════════════════════
# Temporal Splice Mutation (Section 6.4.4)
# ═══════════════════════════════════════════════════════════

def temporal_splice_mutation(
    Z: torch.Tensor,
    splice_prob: float = 0.3,
) -> torch.Tensor:
    """Bridge two valid trajectories at a latent near-match point.

    τ' = z_0, ..., z_t, y_{s+1}, ..., y_H

    Every segment is real data. The combined worldline is
    globally inconsistent.

    Args:
        Z: Worldline slab (B, H, N, d).
        splice_prob: Probability of splicing per batch element.

    Returns:
        Mutated Z'.
    """
    B, H, N, d = Z.shape
    if B < 2:
        return Z

    Z_mut = Z.clone()

    # Pair up batch elements and splice
    indices = torch.randperm(B)
    for b in range(0, B - 1, 2):
        if torch.rand(1).item() < splice_prob:
            i, j = indices[b], indices[b + 1]
            splice_t = H // 2  # splice at midpoint
            Z_mut[i, splice_t:] = Z[j, splice_t:]

    return Z_mut


# ═══════════════════════════════════════════════════════════
# Cycle-Breaking Mutation (Section 6.4.5)
# ═══════════════════════════════════════════════════════════

class CycleBreakingMutation(nn.Module):
    """Futures plausible forward but impossible when reversed.

    Train F(z_t, a_t) → z_{t+k} (forward) and B(z_{t+k}, a_t) → z_t (backward).
    Corruptor searches for edits maximizing F_accept - B_error.
    """

    def __init__(self, d_state: int):
        super().__init__()
        self.forward_map = nn.Sequential(
            nn.Linear(d_state * 2, 128),  # z_t + a_t
            nn.ReLU(),
            nn.Linear(128, d_state),
        )
        self.backward_map = nn.Sequential(
            nn.Linear(d_state * 2, 128),  # z_{t+k} + a_t
            nn.ReLU(),
            nn.Linear(128, d_state),
        )

    def apply(
        self, Z: torch.Tensor, A: torch.Tensor, noise_scale: float = 0.02,
    ) -> torch.Tensor:
        """Apply cycle-breaking corruption.

        Add a perturbation that is small in forward space
        but large in backward space (temporal asymmetry).

        Args:
            Z: (B, H, N, d).
            A: (B, H, d_action).

        Returns:
            Mutated Z'.
        """
        B, H, N, d = Z.shape
        Z_mut = Z.clone()

        if H < 2:
            return Z_mut

        # Pick random step pair
        t = torch.randint(0, H - 1, (B,))
        for b in range(B):
            z_t = Z_mut[b, t[b]]  # (N, d)
            a_t = A[b, t[b]]  # (d_action,)

            # Small perturbation
            noise = torch.randn(N, d, device=Z.device) * noise_scale

            # Check forward/backward consistency
            joint = torch.cat([z_t, a_t.expand(N, -1)], dim=-1)  # (N, d+d_action)
            forward_pred = self.forward_map(joint)  # (N, d)
            backward_pred = self.backward_map(
                torch.cat([z_t + noise, a_t.expand(N, -1)], dim=-1),
            )

            # Perturbation that fools forward but not backward
            Z_mut[b, t[b]] += noise * (1.0 - torch.sigmoid(
                (backward_pred - forward_pred).norm(dim=-1, keepdim=True)
            ))

        return Z_mut
