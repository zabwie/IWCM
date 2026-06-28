"""SlotPermanenceEncoder — video encoder with stable temporal slot identity.

Combines three mechanisms to solve the slot permutation problem:
  1. Learned SlotTransitionPredictor for strong next-frame initialization
  2. SpatialSlotAnchoring as a positional prior in slot attention
  3. ContrastiveContentLoss for direct temporal content consistency

Architecture:
  For each frame t:
    1. CNN backbone → feature map
    2. + Positional encoding
    3. Predict slot init from t-1 via SlotTransitionPredictor
    4. Slot attention with spatial anchoring → slots_t
    5. Accumulate content consistency loss across time

Training losses:
  - L_recon: MSE(frame, decode(slots)) — slot quality
  - L_content: MSE(content_t, content_{t+1}) per matched slot — temporal stability
  - L_diversity: push apart different slots' content within each frame
  - L_iwcm: IWCM energy margin (valid low, invalid high) — constraint learning
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict
import math

from .slot_attention import SlotAttention
from .slot_transition import SlotTransitionPredictor
from ..utils.base import BaseModel


def _make_cnn_backbone(
    frame_size: int = 64,
    in_channels: int = 3,
    cnn_channels: Tuple[int, ...] = (32, 64, 64, 128),
) -> Tuple[nn.Module, int, int]:
    """Build CNN backbone and compute output dimensions.

    Returns:
        cnn: Sequential CNN module.
        cnn_output_dim: Output channels of final layer.
        feature_size: Spatial size of output feature map.
    """
    layers = []
    curr_channels = in_channels
    curr_size = frame_size
    for ch in cnn_channels:
        layers.extend([
            nn.Conv2d(curr_channels, ch, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
        ])
        curr_channels = ch
        curr_size //= 2
    cnn = nn.Sequential(*layers)
    return cnn, cnn_channels[-1], curr_size


class SlotPermanenceEncoder(BaseModel):
    """Video encoder with temporal slot identity preservation.

    Replaces VideoEncoder's per-frame slot attention (which produces
    permuted slots) with a temporally-aware pipeline:
    - Transition model predicts where slots should be next frame
    - Spatial anchoring biases each slot to a preferred region
    - Content consistency loss trains slots to maintain identity across time

    Args:
        frame_size: Width/height of input frames.
        in_channels: Input channels (3 for RGB).
        cnn_channels: CNN layer output channels.
        num_slots: Number of object slots (N).
        slot_dim: Dimension per slot (d).
        content_dim: First content_dim channels treated as invariant content.
        slot_iters: Number of slot attention iterations.
        d_action: Action encoding dimension.
        transition_hidden: Hidden dim for SlotTransitionPredictor.
        anchor_beta: Spatial anchoring strength (higher = stickier).
        anchor_beta_anneal: Anneal target for beta.
        use_gru_transition: Use GRU in transition predictor.
    """

    def __init__(
        self,
        frame_size: int = 64,
        in_channels: int = 3,
        cnn_channels: Tuple[int, ...] = (32, 64, 64, 128),
        num_slots: int = 8,
        slot_dim: int = 64,
        content_dim: int = 32,
        slot_iters: int = 3,
        d_action: int = 11,
        transition_hidden: int = 256,
        anchor_beta: float = 10.0,
        anchor_beta_anneal: float = 2.0,
        use_gru_transition: bool = False,
    ):
        super().__init__()
        self.frame_size = frame_size
        self.num_slots = num_slots
        self.slot_dim = slot_dim
        self.content_dim = content_dim
        self.d_action = d_action

        # CNN backbone
        self.cnn, cnn_output_dim, feature_size = _make_cnn_backbone(
            frame_size, in_channels, cnn_channels
        )
        self.cnn_output_dim = cnn_output_dim
        self.feature_size = feature_size
        n_patches = feature_size * feature_size

        # Positional encoding
        self.pos_embed = nn.Parameter(
            torch.randn(1, n_patches, cnn_output_dim) * 0.02
        )

        # Slot attention with spatial anchoring
        self.slot_attention = SlotAttention(
            num_slots=num_slots,
            slot_dim=slot_dim,
            input_dim=cnn_output_dim,
            num_iters=slot_iters,
            feature_size=feature_size,
            use_spatial_anchor=True,
            anchor_beta=anchor_beta,
            anchor_beta_anneal=anchor_beta_anneal,
        )

        # Transition predictor
        self.transition = SlotTransitionPredictor(
            slot_dim=slot_dim,
            d_action=d_action,
            hidden_dim=transition_hidden,
            use_gru=use_gru_transition,
        )

    def encode_frame(
        self,
        frame: torch.Tensor,
        init_slots: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Encode a single frame to slots via CNN + slot attention.

        Args:
            frame: (B, C, H, W).
            init_slots: Optional (B, N, d) initial slot values.

        Returns:
            Slots of shape (B, N, d).
        """
        B = frame.shape[0]

        # CNN
        features = self.cnn(frame)  # (B, C_out, fH, fW)
        B, C, Hf, Wf = features.shape

        # Flatten spatial + add positional encoding
        features = features.reshape(B, C, Hf * Wf).transpose(1, 2)  # (B, N_patches, C)
        features = features + self.pos_embed

        # Slot attention with spatial anchoring
        slots = self.slot_attention(features, init_slots=init_slots)  # (B, N, d)
        return slots

    def forward(
        self,
        video: torch.Tensor,
        actions: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Encode video with temporal slot propagation and transition prediction.

        Processes frames sequentially: for each frame t > 0, predicts slot
        initialization from frame t-1's slots and action a_{t-1}, then runs
        slot attention with spatial anchoring.

        Args:
            video: (B, H, C, W_frame, H_frame).
            actions: (B, H, d_action) one-hot, (B, H) integer indices, or None.
                If None or integer, zeros/one-hot used.

        Returns:
            Worldline slab of shape (B, H, N, d).
        """
        B, H, C, H_f, W_f = video.shape
        self.transition.reset_state()

        # Normalize actions to (B, H, d_action)
        if actions is None:
            actions_enc = torch.zeros(B, H, self.d_action, device=video.device)
        elif actions.dim() == 2:
            actions_enc = F.one_hot(actions.long(), num_classes=self.d_action).float()
        else:
            actions_enc = actions

        all_slots = []

        # Frame 0: no transition — use learned initial slots (via slot_attention default)
        frame_0 = video[:, 0]
        slots = self.encode_frame(frame_0)
        all_slots.append(slots)

        # Frames 1..H-1: predict init from previous + action, then refine
        for t in range(1, H):
            slots_init = self.transition(slots.detach(), actions_enc[:, t - 1])
            frame_t = video[:, t]
            slots = self.encode_frame(frame_t, init_slots=slots_init)
            all_slots.append(slots)

        return torch.stack(all_slots, dim=1)  # (B, H, N, d)

    def forward_parallel(
        self,
        video: torch.Tensor,
    ) -> torch.Tensor:
        """Encode video with parallel per-frame encoding (no temporal tracking).

        For ablation: same as VideoEncoder.forward() — processes all frames
        independently with no temporal propagation.

        Args:
            video: (B, H, C, W_frame, H_frame).

        Returns:
            Worldline slab of shape (B, H, N, d).
        """
        B, H, C, H_f, W_f = video.shape
        frames = video.reshape(B * H, C, H_f, W_f)
        slots = self.encode_frame(frames)  # (B*H, N, d)
        return slots.reshape(B, H, self.num_slots, self.slot_dim)

    def step_anchor_beta(self, decay: float = 0.999):
        """Anneal spatial anchoring strength.

        Call after each training step.
        """
        self.slot_attention.step_anchor_beta(decay)


# ─── Loss Functions ──────────────────────────────────────────────────────────


def content_smoothness_loss(
    slots: torch.Tensor,
    content_dim: int,
) -> torch.Tensor:
    """Penalize content change for same-index slots across adjacent frames.

    Args:
        slots: (B, H, N, d) — encoded video slots.
        content_dim: Number of leading dimensions treated as invariant content.

    Returns:
        Scalar loss.
    """
    B, H, N, d = slots.shape
    if H < 2:
        return torch.tensor(0.0, device=slots.device)

    content = slots[..., :content_dim]
    content_t = content[:, :-1]
    content_tp1 = content[:, 1:]
    return F.mse_loss(content_t, content_tp1)


def slot_diversity_loss(
    slots: torch.Tensor,
    content_dim: int,
    margin: float = 0.5,
) -> torch.Tensor:
    """Push apart different slots' content within each frame.

    Args:
        slots: (B, H, N, d) — encoded video slots.
        content_dim: Number of leading dimensions treated as invariant content.
        margin: Minimum desired angular distance (0 to 1).

    Returns:
        Scalar loss.
    """
    B, H, N, d = slots.shape

    content = slots[..., :content_dim]
    content_n = F.normalize(content, dim=-1)

    sim = torch.einsum('bhid,bhjd->bhij', content_n, content_n)
    self_mask = 1.0 - torch.eye(N, device=slots.device).unsqueeze(0).unsqueeze(0)
    sim = sim * self_mask

    return F.relu(sim - (1.0 - margin)).mean()


def transition_consistency_loss(
    slots: torch.Tensor,
    transition: SlotTransitionPredictor,
    actions: torch.Tensor,
) -> torch.Tensor:
    """Penalize discrepancy between predicted and actual slots.

    Args:
        slots: (B, H, N, d) — encoded slots.
        transition: SlotTransitionPredictor module.
        actions: (B, H, d_action) one-hot, or (B, H) integer indices.

    Returns:
        Scalar loss.
    """
    B, H, N, d = slots.shape
    if H < 2:
        return torch.tensor(0.0, device=slots.device)

    if actions.dim() == 2:
        d_action = transition.d_action
        actions_enc = F.one_hot(actions.long(), num_classes=d_action).float()
    else:
        actions_enc = actions

    transition.reset_state()
    loss = 0.0
    for t in range(H - 1):
        pred = transition(slots[:, t].detach(), actions_enc[:, t])
        loss += F.smooth_l1_loss(pred, slots[:, t + 1].detach())

    return loss / (H - 1)


def compute_slot_switch_rate(
    slots: torch.Tensor,
    content_dim: int,
) -> float:
    """Measure slot identity switching rate across adjacent frames.

    Args:
        slots: (B, H, N, d).
        content_dim: Number of content dimensions for slot matching.

    Returns:
        Switch rate (0 = perfectly stable, 1 = completely scrambled).
    """
    B, H, N, d = slots.shape
    if H < 2:
        return 0.0

    content = slots[..., :content_dim]
    switches = 0
    total = 0

    with torch.no_grad():
        for b in range(B):
            for t in range(H - 1):
                ct_n = F.normalize(content[b, t], dim=-1)
                ct1_n = F.normalize(content[b, t + 1], dim=-1)
                sim = ct_n @ ct1_n.T
                best_matches = sim.argmax(dim=-1)
                ideal = torch.arange(N, device=slots.device)
                switches += (best_matches != ideal).sum().item()
                total += N

    return switches / max(total, 1)
