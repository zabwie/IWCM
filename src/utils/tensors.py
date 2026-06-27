"""Tensor utilities for worldline slab operations.

Worldline slab shape: (B, H, N, d)
  B = batch size
  H = horizon length
  N = number of object slots
  d = latent dimension per slot
"""

import torch
from typing import Tuple


def worldline_to_sequence(slab: torch.Tensor) -> torch.Tensor:
    """Flatten worldline slab to sequence: (B, H, N, d) → (B*H, N, d).

    Args:
        slab: Tensor of shape (B, H, N, d).

    Returns:
        Tensor of shape (B*H, N, d).
    """
    B, H, N, d = slab.shape
    return slab.reshape(B * H, N, d)


def sequence_to_worldline(seq: torch.Tensor, H: int) -> torch.Tensor:
    """Reshape sequence back to worldline slab: (B*H, N, d) → (B, H, N, d).

    Args:
        seq: Tensor of shape (B*H, N, d).
        H: Horizon length.

    Returns:
        Tensor of shape (B, H, N, d).
    """
    BH, N, d = seq.shape
    B = BH // H
    return seq.reshape(B, H, N, d)


def normalize_worldline(
    slab: torch.Tensor, dim: int = -1, eps: float = 1e-8
) -> torch.Tensor:
    """L2-normalize over specified dimension.

    Args:
        slab: Input tensor of any shape.
        dim: Dimension to normalize over (default: last).
        eps: Small epsilon for numerical stability.

    Returns:
        L2-normalized tensor of same shape.
    """
    norm = slab.norm(p=2, dim=dim, keepdim=True)
    return slab / (norm + eps)


def compute_slab_distances(
    slab1: torch.Tensor, slab2: torch.Tensor
) -> torch.Tensor:
    """Per-timestep MSE between two worldlines.

    Computes ||slab1[:, t] - slab2[:, t]||² / (N*d) for each t.

    Args:
        slab1: Tensor of shape (B, H, N, d).
        slab2: Tensor of shape (B, H, N, d).

    Returns:
        Tensor of shape (B, H) with per-timestep mean squared error.
    """
    diff = slab1 - slab2
    return diff.pow(2).mean(dim=(-1, -2))  # mean over N and d


def compute_slab_cosine(
    slab1: torch.Tensor, slab2: torch.Tensor
) -> torch.Tensor:
    """Per-timestep cosine similarity between two worldlines.

    Args:
        slab1: Tensor of shape (B, H, N, d).
        slab2: Tensor of shape (B, H, N, d).

    Returns:
        Tensor of shape (B, H) with cosine similarities.
    """
    flat1 = slab1.reshape(*slab1.shape[:2], -1)  # (B, H, N*d)
    flat2 = slab2.reshape(*slab2.shape[:2], -1)
    cos = torch.nn.functional.cosine_similarity(flat1, flat2, dim=-1)
    return cos


def causal_mask(h: int, device: torch.device = None) -> torch.Tensor:
    """Create lower-triangular causal mask of size (h, h).

    Used for autoregressive attention masking.

    Args:
        h: Sequence length.
        device: Torch device.

    Returns:
        Boolean mask of shape (h, h) where mask[i, j] = True if j <= i.
    """
    mask = torch.tril(torch.ones(h, h, dtype=torch.bool, device=device))
    return mask


def build_adjacency_mask(
    h: int, window: int = 1, device: torch.device = None
) -> torch.Tensor:
    """Create adjacency mask for local transition constraints.

    Mask[i, j] = True if |i - j| <= window.

    Args:
        h: Sequence length.
        window: Maximum distance for adjacency (default 1 = pairs).
        device: Torch device.

    Returns:
        Boolean mask of shape (h, h).
    """
    idx = torch.arange(h, device=device)
    diff = (idx.unsqueeze(0) - idx.unsqueeze(1)).abs()
    return diff <= window


def split_slots_content_pose(
    slab: torch.Tensor, content_dim: int
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Split slot representation into content, pose, hidden components.

    Per TAMG Section 6.4: z = [c, p, h].

    Args:
        slab: Tensor of shape (..., N, d).
        content_dim: Dimension of content subspace.

    Returns:
        Tuple of (content, pose, hidden) tensors.
    """
    total_dim = slab.shape[-1]
    assert content_dim <= total_dim, f"content_dim {content_dim} > total_dim {total_dim}"
    pose_dim = (total_dim - content_dim) // 2
    hidden_dim = total_dim - content_dim - pose_dim

    content = slab[..., :content_dim]
    pose = slab[..., content_dim:content_dim + pose_dim]
    hidden = slab[..., content_dim + pose_dim:]

    return content, pose, hidden


def compose_slots_content_pose(
    content: torch.Tensor, pose: torch.Tensor, hidden: torch.Tensor
) -> torch.Tensor:
    """Compose content, pose, hidden back into single slot representation.

    Args:
        content: Tensor of shape (..., N, c_dim).
        pose: Tensor of shape (..., N, p_dim).
        hidden: Tensor of shape (..., N, h_dim).

    Returns:
        Tensor of shape (..., N, c_dim + p_dim + h_dim).
    """
    return torch.cat([content, pose, hidden], dim=-1)
