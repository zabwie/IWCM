"""Spatial anchoring utilities for slot attention.

Provides position grid generation and anchor initialization for
spatially-grounded slot attention, where each slot has a learned
preferred spatial region that biases its attention.

The core idea: slot attention is permutation-invariant, causing
random slot-object assignments across frames. Adding a learned
spatial prior per slot naturally assigns slots to consistent
spatial regions, which maps to consistent object assignments
when objects move smoothly.
"""

import torch
import torch.nn as nn
import math
from typing import Tuple


def build_position_grid(feature_size: int, device: torch.device) -> torch.Tensor:
    """Build a normalized position grid for CNN feature map patches.

    For a feature_size × feature_size spatial grid, returns
    the (x, y) center coordinates of each patch, normalized to [0, 1].

    Args:
        feature_size: Spatial dimension of the feature map (e.g., 4 for 4×4).
        device: Target device.

    Returns:
        Position grid of shape (1, feature_size^2, 2).
    """
    # Create meshgrid of normalized positions
    coords = torch.linspace(0.5 / feature_size, 1.0 - 0.5 / feature_size, feature_size, device=device)
    gy, gx = torch.meshgrid(coords, coords, indexing='ij')
    positions = torch.stack([gx.flatten(), gy.flatten()], dim=-1)  # (feature_size^2, 2)
    return positions.unsqueeze(0)  # (1, N_patches, 2)


def init_anchor_grid(num_slots: int, feature_size: int, device: torch.device) -> torch.Tensor:
    """Initialize anchor positions in a regular grid over the feature map.

    Distributes num_slots anchors evenly over the feature_size × feature_size
    spatial grid. For example, 8 slots on a 4×4 grid get placed at 8 positions.

    Uses jittered grid: anchors are centered in grid cells with small random
    offset to break symmetry.

    Args:
        num_slots: Number of slots (K).
        feature_size: Spatial dimension of feature map.
        device: Target device.

    Returns:
        Anchor positions of shape (num_slots, 2), normalized to [0, 1].
    """
    # Determine grid layout: ceil(sqrt(K)) × ceil(sqrt(K))
    grid_dim = math.ceil(math.sqrt(num_slots))
    cell_size = 1.0 / grid_dim

    anchors = torch.zeros(num_slots, 2, device=device)
    for k in range(num_slots):
        row = k // grid_dim
        col = k % grid_dim
        # Center of cell + small jitter
        anchors[k, 0] = (col + 0.5) * cell_size + torch.randn(1, device=device).item() * cell_size * 0.1
        anchors[k, 1] = (row + 0.5) * cell_size + torch.randn(1, device=device).item() * cell_size * 0.1
        # Clamp to [0, 1]
        anchors[k].clamp_(0.0, 1.0)

    return anchors


def compute_spatial_attention_bias(
    anchor_positions: torch.Tensor,
    position_grid: torch.Tensor,
    beta: float = 10.0,
) -> torch.Tensor:
    """Compute spatial bias for slot attention logits.

    Each slot's attention to each feature position is biased by
    a Gaussian centered at the slot's anchor:

        bias[k, n] = -beta * ||pos_grid[n] - anchor[k]||^2

    Higher beta = stronger spatial anchoring (slots stick closer to anchors).
    During training, beta can be annealed from high to low.

    Args:
        anchor_positions: (K, 2) — learned anchor positions per slot.
        position_grid: (1, N_patches, 2) — positions of input features.
        beta: Anchoring strength (default 10.0).

    Returns:
        Bias tensor of shape (1, K, N_patches) for addition to attention logits.
    """
    K = anchor_positions.shape[0]
    N = position_grid.shape[1]

    # Compute squared distances: (K, N)
    # anchors: (K, 2) → (K, 1, 2), grid: (1, N, 2) → (1, N, 2)
    diff = anchor_positions.unsqueeze(1) - position_grid  # (K, N, 2)
    dist_sq = (diff ** 2).sum(dim=-1)  # (K, N)

    # Negative Gaussian bias (unnormalized)
    bias = -beta * dist_sq  # (K, N)

    return bias.unsqueeze(0)  # (1, K, N)


class SpatialAnchorAttention(nn.Module):
    """Wrapper that adds spatial anchoring to an existing SlotAttention module.

    This is an alternative to modifying SlotAttention directly. It pre-computes
    the spatial bias and passes it to a modified attention computation.

    Currently unused — spatial anchoring is integrated directly into SlotAttention.
    Kept as a reference for standalone anchoring.
    """

    def __init__(
        self,
        num_slots: int,
        feature_size: int,
        beta: float = 10.0,
    ):
        super().__init__()
        self.num_slots = num_slots
        self.feature_size = feature_size
        self.beta = beta

        # Register position grid (non-trainable)
        pos_grid = build_position_grid(feature_size, torch.device('cpu'))
        self.register_buffer('position_grid', pos_grid)

        # Learnable anchor positions
        anchors = init_anchor_grid(num_slots, feature_size, torch.device('cpu'))
        self.anchor_positions = nn.Parameter(anchors)

    def forward(self) -> torch.Tensor:
        """Compute the spatial attention bias.

        Returns:
            Bias of shape (1, num_slots, feature_size^2).
        """
        return compute_spatial_attention_bias(
            self.anchor_positions, self.position_grid, self.beta,
        )
