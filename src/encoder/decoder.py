"""Video decoder: reconstructs frames from slot-based latents.

Spatial broadcast decoder that reconstructs each frame from its
slot representations using alpha compositing.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from ..utils.base import BaseModel


class VideoDecoder(BaseModel):
    """Decodes slot representations back into video frames.

    Each slot independently reconstructs a full frame (spatial broadcast),
    then alpha compositing blends them.

    Args:
        slot_dim: Dimension of each slot.
        frame_size: Target frame width/height.
        out_channels: Number of output channels (3 for RGB).
        cnn_channels: Channels for upsampling (reversed from encoder).
    """

    def __init__(
        self,
        slot_dim: int = 64,
        frame_size: int = 64,
        out_channels: int = 3,
        cnn_channels: tuple = (128, 64, 64, 32, 16),
    ):
        super().__init__()
        self.slot_dim = slot_dim
        self.frame_size = frame_size
        self.out_channels = out_channels

        self.init_size = frame_size // (2 ** (len(cnn_channels) - 1))

        # Broadcast MLP: slot → initial feature map
        broadcast_dim = cnn_channels[0] * self.init_size * self.init_size
        self.broadcast = nn.Sequential(
            nn.Linear(slot_dim, 256),
            nn.ReLU(),
            nn.Linear(256, broadcast_dim),
        )

        # CNN upsampling (transposed convolutions)
        up_layers = []
        curr_ch = cnn_channels[0]
        for ch in cnn_channels[1:]:
            up_layers.extend([
                nn.ConvTranspose2d(curr_ch, ch, kernel_size=4, stride=2, padding=1),
                nn.ReLU(inplace=True),
            ])
            curr_ch = ch
        up_layers.append(
            nn.ConvTranspose2d(curr_ch, out_channels + 1, kernel_size=3, stride=1, padding=1)
        )  # +1 for alpha
        self.upsample = nn.Sequential(*up_layers)

    def forward(
        self, slots: torch.Tensor
    ) -> torch.Tensor:
        """Decode slots to frames.

        Args:
            slots: Tensor of shape (..., num_slots, slot_dim).

        Returns:
            Frames of shape (..., out_channels, frame_size, frame_size).
        """
        *batch_dims, K, d = slots.shape
        flat_batch = 1
        for s in batch_dims:
            flat_batch *= s
        slots_flat = slots.reshape(flat_batch, K, d)

        # Broadcast each slot to a feature map
        broadcasted = self.broadcast(slots_flat)  # (flat_batch * K, broadcast_dim)
        broadcasted = broadcasted.reshape(
            flat_batch * K, -1, self.init_size, self.init_size
        )

        # Upsample
        raw = self.upsample(broadcasted)  # (flat_batch * K, out_channels+1, frame_size, frame_size)

        # Reshape per batch and slot
        raw = raw.reshape(flat_batch, K, self.out_channels + 1, self.frame_size, self.frame_size)

        # Split into RGB and alpha
        rgb = raw[:, :, :self.out_channels]  # (flat_batch, K, out_channels, H, W)
        alpha = raw[:, :, self.out_channels:self.out_channels + 1]  # (flat_batch, K, 1, H, W)

        # Alpha compositing
        alpha = F.softmax(alpha, dim=1)  # normalize alphas across slots
        reconstructed = (rgb * alpha).sum(dim=1)  # (flat_batch, out_channels, H, W)

        # Reshape back to batch dimensions
        reconstructed = reconstructed.reshape(*batch_dims, self.out_channels, self.frame_size, self.frame_size)

        return reconstructed

    def decode_frame(
        self, slots: torch.Tensor
    ) -> torch.Tensor:
        """Decode a single frame's slots.

        Args:
            slots: (num_slots, slot_dim) or (B, num_slots, slot_dim).

        Returns:
            Frame of shape (out_channels, frame_size, frame_size).
        """
        if slots.dim() == 2:
            slots = slots.unsqueeze(0)
        return self.forward(slots)
