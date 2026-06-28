"""Slot Attention module for object-centric latent decomposition.

Implements iterative slot attention (Locatello et al. 2020) for
decomposing CNN features into object-centric slot representations.

Extended with optional spatial anchoring: each slot can have a learned
preferred spatial region that biases its attention, reducing slot
permutation across frames in video settings.

Used by the video encoder for TAMG (Experiment 2).

Optimized for GPU: uses efficient matrix operations and avoids
unnecessary CPU transfers.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import math


class SlotAttention(nn.Module):
    """Iterative slot attention for object-centric representation.

    Given input features of shape (B, N_features, d_input), learns
    K slots of shape (B, K, d_slot) that compete to explain the input.

    When spatial anchoring is enabled, each slot is biased toward a specific
    spatial region via a learnable anchor position. The bias is added to
    attention logits before softmax:

        dots = q @ k^T / sqrt(d) + spatial_bias
        spatial_bias[k, n] = -beta * ||pos_grid[n] - anchor[k]||^2

    This encourages each slot to consistently attend to the same spatial
    region across frames, providing stable slot-object correspondence.

    Args:
        num_slots: Number of object slots K.
        slot_dim: Dimension of each slot d_slot.
        input_dim: Dimension of input features d_input.
        num_iters: Number of iterative refinement steps.
        eps: Small constant for numerical stability.
        feature_size: Spatial size of CNN feature map (e.g., 4 for 4×4).
            Required only when use_spatial_anchor=True.
        use_spatial_anchor: Enable spatial anchoring (default False).
        anchor_beta: Strength of spatial bias (higher = stronger anchoring).
        anchor_beta_anneal: If > 0, beta starts at anchor_beta and decays
            to this value over training via exponential schedule.
    """

    def __init__(
        self,
        num_slots: int = 6,
        slot_dim: int = 64,
        input_dim: int = 128,
        num_iters: int = 3,
        eps: float = 1e-8,
        feature_size: int = 4,
        use_spatial_anchor: bool = False,
        anchor_beta: float = 10.0,
        anchor_beta_anneal: float = 0.0,
    ):
        super().__init__()
        self.num_slots = num_slots
        self.slot_dim = slot_dim
        self.input_dim = input_dim
        self.num_iters = num_iters
        self.eps = eps
        self.use_spatial_anchor = use_spatial_anchor

        # Input projections to key, query space
        self.key_proj = nn.Linear(input_dim, slot_dim)
        self.value_proj = nn.Linear(input_dim, slot_dim)

        # Slot query projection
        self.query_proj = nn.Linear(slot_dim, slot_dim)

        # GRU for slot updates
        self.gru = nn.GRUCell(slot_dim, slot_dim)

        # Per-iteration MLP residual
        self.mlp = nn.Sequential(
            nn.Linear(slot_dim, slot_dim * 4),
            nn.ReLU(),
            nn.Linear(slot_dim * 4, slot_dim),
        )
        self.norm1 = nn.LayerNorm(slot_dim)
        self.norm2 = nn.LayerNorm(slot_dim)

        # Learnable initial slots (broadcast per batch)
        self.slots_mu = nn.Parameter(torch.randn(1, 1, slot_dim) * 0.02)
        self.slots_logsigma = nn.Parameter(torch.zeros(1, 1, slot_dim))

        # ── Spatial anchoring ──────────────────────────────────────────
        if use_spatial_anchor:
            self.feature_size = feature_size
            self.anchor_beta_init = anchor_beta
            self.anchor_beta_anneal = anchor_beta_anneal
            self.register_buffer('anchor_beta_current', torch.tensor(float(anchor_beta)))

            # Position grid: (1, feature_size^2, 2) with normalized (x,y) coords
            n_patches = feature_size * feature_size
            coords = torch.linspace(0.5 / feature_size, 1.0 - 0.5 / feature_size, feature_size)
            gy, gx = torch.meshgrid(coords, coords, indexing='ij')
            pos_grid = torch.stack([gx.flatten(), gy.flatten()], dim=-1).unsqueeze(0)  # (1, N_patches, 2)
            self.register_buffer('position_grid', pos_grid)

            # Learnable anchor positions: (num_slots, 2)
            # Initialized in a grid pattern over the feature map
            grid_dim = math.ceil(math.sqrt(num_slots))
            cell_size = 1.0 / grid_dim
            anchors = torch.zeros(num_slots, 2)
            for k in range(num_slots):
                row, col = k // grid_dim, k % grid_dim
                anchors[k, 0] = (col + 0.5) * cell_size
                anchors[k, 1] = (row + 0.5) * cell_size
            self.anchor_positions = nn.Parameter(anchors)
            # Clamp anchors to valid range
            self.register_buffer('anchor_clamp_min', torch.tensor(0.02))
            self.register_buffer('anchor_clamp_max', torch.tensor(0.98))

    def _compute_spatial_bias(self, device: torch.device) -> torch.Tensor:
        """Compute spatial attention bias from anchor positions.

        Returns:
            Bias of shape (1, K, N_patches) added to attention logits.
        """
        K = self.num_slots
        N = self.position_grid.shape[1]
        beta = self.anchor_beta_current

        # anchors: (K, 2) → (K, 1, 2), grid: (1, N, 2)
        diff = self.anchor_positions.unsqueeze(1) - self.position_grid.to(device)  # (K, N, 2)
        dist_sq = (diff ** 2).sum(dim=-1)  # (K, N)

        # Negative Gaussian bias (unnormalized):
        # Strong negative bias far from anchor → slot avoids those regions
        bias = -beta * dist_sq  # (K, N)

        return bias.unsqueeze(0)  # (1, K, N)

    def step_anchor_beta(self, decay: float = 0.999):
        """Anneal anchor beta toward the minimum value.

        Call after each training step to gradually reduce spatial anchoring
        strength, allowing slots to refine their attention patterns.

        Args:
            decay: Exponential decay factor per step.
        """
        if self.use_spatial_anchor and self.anchor_beta_anneal > 0:
            current = self.anchor_beta_current.item()
            target = self.anchor_beta_anneal
            new_val = target + (current - target) * decay
            self.anchor_beta_current.fill_(new_val)

    def _clamp_anchors(self):
        """Clamp anchor positions to valid range [0.02, 0.98]."""
        if self.use_spatial_anchor:
            with torch.no_grad():
                self.anchor_positions.clamp_(
                    self.anchor_clamp_min, self.anchor_clamp_max
                )

    def forward(
        self, inputs: torch.Tensor, init_slots: torch.Tensor = None
    ) -> torch.Tensor:
        """Apply slot attention to input features.

        Args:
            inputs: Feature tensor of shape (B, N_features, d_input).
            init_slots: Optional (B, num_slots, slot_dim) — if provided,
                initializes slots from these instead of learned parameters.
                Enables temporal slot tracking across frames.

        Returns:
            Slots tensor of shape (B, num_slots, slot_dim).
        """
        B = inputs.shape[0]
        device = inputs.device

        # Project inputs to key/value space
        k = self.key_proj(inputs)    # (B, N_feat, slot_dim)
        v = self.value_proj(inputs)  # (B, N_feat, slot_dim)

        # Initialize slots
        if init_slots is not None:
            slots = init_slots
        else:
            slots = self.slots_mu + torch.exp(self.slots_logsigma) * torch.randn(
                B, self.num_slots, self.slot_dim, device=device
            )

        # Pre-compute spatial bias (constant across iterations)
        spatial_bias = None
        if self.use_spatial_anchor:
            spatial_bias = self._compute_spatial_bias(device)  # (1, K, N_feat)

        # Iterative refinement
        for _ in range(self.num_iters):
            slots_prev = slots

            # Query from current slots
            q = self.query_proj(slots)  # (B, K, slot_dim)

            # Attention: slots attend to inputs
            dots = torch.einsum("bkd,bnd->bkn", q, k)  # (B, K, N_feat)
            dots = dots / (self.slot_dim ** 0.5)

            # Apply spatial anchoring bias
            if spatial_bias is not None:
                dots = dots + spatial_bias  # (B, K, N_feat)

            attn = F.softmax(dots, dim=1)  # normalize over slots: (B, K, N_feat)

            # Normalize over inputs (weighted mean)
            attn_sum = attn.sum(dim=-1, keepdim=True) + self.eps
            attn = attn / attn_sum

            # Update: weighted sum of values
            updates = torch.einsum("bkn,bnd->bkd", attn, v)  # (B, K, slot_dim)

            # GRU update
            slots = self.gru(
                updates.reshape(B * self.num_slots, self.slot_dim),
                slots_prev.reshape(B * self.num_slots, self.slot_dim),
            ).reshape(B, self.num_slots, self.slot_dim)

            # Residual MLP
            slots = slots + self.mlp(self.norm1(slots))
            slots = self.norm2(slots)

        # Clamp anchor positions after each forward (guard against drift)
        if self.use_spatial_anchor:
            self._clamp_anchors()

        return slots
