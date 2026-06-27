"""Video encoder: CNN backbone + slot attention → object-centric latents.

Encodes rendered grid world frames into (B, H, N, d) worldline slabs
for the TAMG pipeline (Experiment 2).
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional

from .slot_attention import SlotAttention
from ..utils.base import BaseModel


class VideoEncoder(BaseModel):
    """Encodes video frames into object-centric latent worldlines.

    Architecture:
      1. CNN backbone (per-frame) → feature maps
      2. Flatten spatial → (B, N_patches, d_cnn)
      3. Positional encoding
      4. Slot attention (per-frame) → (B, N_slots, d_slot)
      5. Stack over time → (B, H, N_slots, d_slot)

    Args:
        frame_size: Width/height of input frames.
        cnn_channels: List of output channels per CNN layer.
        num_slots: Number of object slots.
        slot_dim: Dimension per slot.
        slot_iters: Number of slot attention iterations.
    """

    def __init__(
        self,
        frame_size: int = 64,
        in_channels: int = 3,
        cnn_channels: Tuple[int, ...] = (32, 64, 64, 128),
        num_slots: int = 6,
        slot_dim: int = 64,
        slot_iters: int = 3,
    ):
        super().__init__()
        self.frame_size = frame_size
        self.num_slots = num_slots
        self.slot_dim = slot_dim

        # CNN backbone
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
        self.cnn = nn.Sequential(*layers)
        self.cnn_output_dim = cnn_channels[-1]
        self.feature_size = curr_size  # spatial grid size after CNN

        # Positional encoding — squeezed to 3D (1, N_patches, C)
        n_patches = self.feature_size * self.feature_size
        self.pos_embed = nn.Parameter(
            torch.randn(1, n_patches, self.cnn_output_dim) * 0.02
        )

        # Slot attention
        self.slot_attention = SlotAttention(
            num_slots=num_slots,
            slot_dim=slot_dim,
            input_dim=self.cnn_output_dim,
            num_iters=slot_iters,
        )

    def encode_frame(self, frame: torch.Tensor, init_slots: torch.Tensor = None) -> torch.Tensor:
        """Encode a single frame (or batch of frames) to slots.

        Args:
            frame: Tensor of shape (B, C, H, W) or (B*H, C, H, W).
            init_slots: Optional (B, num_slots, slot_dim) initial slots for tracking.

        Returns:
            Slots of shape (B, num_slots, slot_dim).
        """
        B = frame.shape[0]

        # CNN
        features = self.cnn(frame)  # (B, cnn_dim, feat_size, feat_size)
        B, C, Hf, Wf = features.shape

        # Flatten spatial
        features = features.reshape(B, C, Hf * Wf).transpose(1, 2)  # (B, N_patches, C)

        # Add positional encoding
        features = features + self.pos_embed

        # Slot attention
        slots = self.slot_attention(features, init_slots=init_slots)  # (B, num_slots, slot_dim)

        return slots

    def forward(
        self, video: torch.Tensor
    ) -> torch.Tensor:
        """Encode a video into a latent worldline slab.

        Args:
            video: Tensor of shape (B, H, C, W, H) — batch of frame sequences.

        Returns:
            Worldline slab of shape (B, H, num_slots, slot_dim).
        """
        B, H, C, W_f, H_f = video.shape

        # Process all frames in parallel by flattening H into batch
        frames = video.reshape(B * H, C, W_f, H_f)
        slots = self.encode_frame(frames)  # (B*H, num_slots, slot_dim)

        # Reshape back to worldline
        slots = slots.reshape(B, H, self.num_slots, self.slot_dim)
        return slots

    def forward_temporal(self, video: torch.Tensor) -> torch.Tensor:
        """Encode with temporal slot propagation — t+1 initialized from t.

        Processes frames sequentially, feeding each frame's slot output
        as the next frame's slot initialization. This naturally creates
        temporally consistent slot assignments.

        Args:
            video: (B, H, C, W, H_f) — batch of frame sequences.

        Returns:
            Worldline slab of shape (B, H, num_slots, slot_dim).
        """
        B, H, C, W_f, H_f = video.shape
        all_slots = []

        # First frame: no initialization (uses learned initial slots)
        frame_0 = video[:, 0]
        slots = self.encode_frame(frame_0)
        all_slots.append(slots)

        # Subsequent frames: initialize from previous output
        for t in range(1, H):
            frame_t = video[:, t]
            slots = self.encode_frame(frame_t, init_slots=slots.detach())
            all_slots.append(slots)

        return torch.stack(all_slots, dim=1)  # (B, H, N, d)
        """Encode a single frame (no time dimension).

        Args:
            frame: (C, H, W) or (B, C, H, W).

        Returns:
            Slots of shape (num_slots, slot_dim) or (B, num_slots, slot_dim).
        """
        if frame.dim() == 3:
            frame = frame.unsqueeze(0)
        return self.encode_frame(frame)
