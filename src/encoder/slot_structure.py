"""Slot structure prediction heads for weak oracle supervision.

SlotStructureHead adds lightweight auxiliary prediction heads on top of
learned slot representations, predicting interpretable properties:
  - Position (x, y) — regression
  - Object type — classification (agent, key, door, box, occluder, empty)
  - Activity — whether this slot represents a real object

These heads force the encoder to encode structured information (position,
type, existence) in its slot features — the same signals the IWCM needs
to detect causal violations. Without them, learned slots optimize for
reconstruction + smoothness but don't encode causal structure.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple
from ..utils.base import BaseModel


class SlotStructureHead(BaseModel):
    """Predict structured slot properties from learned slot representations."""

    def __init__(self, slot_dim: int = 64, hidden_dim: int = 64):
        super().__init__()
        self.slot_dim = slot_dim

        self.pos_head = nn.Sequential(
            nn.Linear(slot_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 2))

        self.type_head = nn.Sequential(
            nn.Linear(slot_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 6))

        self.active_head = nn.Sequential(
            nn.Linear(slot_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1))

    def forward(self, slots: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        pos = torch.sigmoid(self.pos_head(slots))
        type_logits = self.type_head(slots)
        active_logits = self.active_head(slots)
        return pos, type_logits, active_logits


def structure_loss(
    slots: torch.Tensor,
    oracle_slots: torch.Tensor,
    structure_head: SlotStructureHead,
    pos_weight: float = 1.0,
    type_weight: float = 0.5,
    active_weight: float = 0.3,
) -> Tuple[torch.Tensor, dict]:
    """Compute structure prediction losses against oracle targets.

    Args:
        slots: (B, N, slot_dim) — learned slot representations.
        oracle_slots: (B, N, oracle_dim) — oracle slot targets.
        structure_head: SlotStructureHead module.
        pos_weight, type_weight, active_weight: Loss component weights.

    Returns:
        Total structure loss and per-component dict.
    """
    B, N, d = slots.shape

    pos_pred, type_logits, active_logits = structure_head(slots)

    pos_target = oracle_slots[:, :, 5:7]
    loss_pos = F.mse_loss(pos_pred, pos_target)

    type_scores = oracle_slots[:, :, :5]
    type_target = type_scores.argmax(dim=-1)
    type_target_valid = type_scores.sum(dim=-1) > 0.3
    type_target = torch.where(type_target_valid, type_target,
                               torch.full_like(type_target, 5))
    loss_type = F.cross_entropy(
        type_logits.reshape(B * N, 6), type_target.reshape(B * N))

    active_target = (type_scores.max(dim=-1).values > 0.3).float()
    loss_active = F.binary_cross_entropy_with_logits(
        active_logits.squeeze(-1), active_target)

    total = pos_weight * loss_pos + type_weight * loss_type + active_weight * loss_active
    return total, {
        "loss_pos": loss_pos.item(),
        "loss_type": loss_type.item(),
        "loss_active": loss_active.item(),
    }


def extract_oracle_targets(batch_oracle, horizon, device):
    """Extract oracle slot targets for all timesteps from a trajectory batch.

    Args:
        batch_oracle: list of (z0, A, Z) tuples from oracle_encode.
        horizon: number of timesteps to extract.
        device: target device.

    Returns:
        oracle_targets: (B, H, N, oracle_dim) — oracle slots for all timesteps.
    """
    B = len(batch_oracle)
    _, _, Z_sample = batch_oracle[0]
    N, oracle_dim = Z_sample.shape[1], Z_sample.shape[2]

    targets = torch.zeros(B, horizon, N, oracle_dim, device=device)
    for b, (z0, A, Z) in enumerate(batch_oracle):
        H_use = min(Z.shape[0], horizon)
        targets[b, :H_use] = torch.from_numpy(Z[:H_use]).float().to(device)

    return targets
