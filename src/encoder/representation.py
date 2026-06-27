"""Object-centric latent representation utilities.

Implements slot decomposition z = [c, p, h] (content, pose, hidden)
as described in TAMG Section 6.4.
"""

import torch
import torch.nn as nn
from typing import Tuple

from ..utils.base import BaseModel


class SlotRepresentation(BaseModel):
    """Structured slot representation with content/pose/hidden decomposition.

    Each slot z ∈ R^d is decomposed into:
      - content (c) ∈ R^c_dim: invariant identity information
      - pose (p) ∈ R^p_dim: position, motion, spatial relations
      - hidden (h) ∈ R^h_dim: occlusion state, uncertainty

    TAMG mutation operators target specific subspaces for corruption.
    """

    def __init__(
        self,
        slot_dim: int = 64,
        content_dim: int = 32,
        pose_dim: int = 16,
    ):
        super().__init__()
        self.slot_dim = slot_dim
        self.content_dim = content_dim
        self.pose_dim = pose_dim
        self.hidden_dim = slot_dim - content_dim - pose_dim

        assert self.hidden_dim > 0, (
            f"slot_dim ({slot_dim}) must be > content_dim ({content_dim}) + pose_dim ({pose_dim})"
        )

    def decompose(
        self, slots: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Decompose slots into content, pose, hidden.

        Args:
            slots: (..., slot_dim).

        Returns:
            content: (..., content_dim)
            pose: (..., pose_dim)
            hidden: (..., hidden_dim)
        """
        content = slots[..., :self.content_dim]
        pose = slots[..., self.content_dim:self.content_dim + self.pose_dim]
        hidden = slots[..., self.content_dim + self.pose_dim:]
        return content, pose, hidden

    def compose(
        self,
        content: torch.Tensor,
        pose: torch.Tensor,
        hidden: torch.Tensor,
    ) -> torch.Tensor:
        """Compose content, pose, hidden into a slot.

        Args:
            content: (..., content_dim).
            pose: (..., pose_dim).
            hidden: (..., hidden_dim).

        Returns:
            Slot: (..., slot_dim).
        """
        return torch.cat([content, pose, hidden], dim=-1)

    def content_distance(
        self, slots1: torch.Tensor, slots2: torch.Tensor
    ) -> torch.Tensor:
        """Compute content-space distance between two sets of slots.

        Args:
            slots1: (..., slot_dim).
            slots2: (..., slot_dim).

        Returns:
            Distance per slot in content space, shape (...).
        """
        c1, _, _ = self.decompose(slots1)
        c2, _, _ = self.decompose(slots2)
        return (c1 - c2).pow(2).sum(dim=-1)

    def pose_distance(
        self, slots1: torch.Tensor, slots2: torch.Tensor
    ) -> torch.Tensor:
        """Compute pose-space distance between two sets of slots.

        Args:
            slots1: (..., slot_dim).
            slots2: (..., slot_dim).

        Returns:
            Distance per slot in pose space, shape (...).
        """
        _, p1, _ = self.decompose(slots1)
        _, p2, _ = self.decompose(slots2)
        return (p1 - p2).pow(2).sum(dim=-1)
