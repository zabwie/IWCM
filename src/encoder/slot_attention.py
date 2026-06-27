"""Slot Attention module for object-centric latent decomposition.

Implements iterative slot attention (Locatello et al. 2020) for
decomposing CNN features into object-centric slot representations.

Used by the video encoder for TAMG (Experiment 2).

Optimized for GPU: uses efficient matrix operations and avoids
unnecessary CPU transfers.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class SlotAttention(nn.Module):
    """Iterative slot attention for object-centric representation.

    Given input features of shape (B, N_features, d_input), learns
    K slots of shape (B, K, d_slot) that compete to explain the input.

    Args:
        num_slots: Number of object slots K.
        slot_dim: Dimension of each slot d_slot.
        input_dim: Dimension of input features d_input.
        num_iters: Number of iterative refinement steps.
        eps: Small constant for numerical stability.
    """

    def __init__(
        self,
        num_slots: int = 6,
        slot_dim: int = 64,
        input_dim: int = 128,
        num_iters: int = 3,
        eps: float = 1e-8,
    ):
        super().__init__()
        self.num_slots = num_slots
        self.slot_dim = slot_dim
        self.input_dim = input_dim
        self.num_iters = num_iters
        self.eps = eps

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

        # Iterative refinement
        for _ in range(self.num_iters):
            slots_prev = slots

            # Query from current slots
            q = self.query_proj(slots)  # (B, K, slot_dim)

            # Attention: slots attend to inputs
            dots = torch.einsum("bkd,bnd->bkn", q, k)  # (B, K, N_feat)
            dots = dots / (self.slot_dim ** 0.5)
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

        return slots
