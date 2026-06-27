"""Global invariant constraint head.

Implements C_invariant(Z) — penalizes violations of global invariants:
object identity, count conservation, ownership, causal ordering, containment.

Uses full self-attention over the entire worldline to detect patterns
that violate global consistency.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .base import ConstraintHead


class InvariantHead(ConstraintHead):
    """Global invariant constraint: objects must maintain identity and count.

    Full self-attention over the worldline detects invariant violations.
    Action conditioning is optional (invariants are mostly action-independent).
    """

    def __init__(
        self,
        d_state: int,
        d_action: int = 11,
        hidden_dim: int = 256,
        num_layers: int = 2,
        num_heads: int = 4,
    ):
        super().__init__(d_state, d_action, hidden_dim)

        self.state_proj = nn.Linear(d_state, hidden_dim)

        # Transformer encoder over the full worldline
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Pool and score
        self.scorer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(
        self, z0: torch.Tensor, A: torch.Tensor, Z: torch.Tensor
    ) -> torch.Tensor:
        B, H, d = Z.shape

        # Project all states to hidden space
        Z_hidden = self.state_proj(Z)  # (B, H, hidden_dim)

        # Full self-attention over the worldline
        Z_encoded = self.transformer(Z_hidden)  # (B, H, hidden_dim)

        # Pool over time (mean pooling)
        Z_pooled = Z_encoded.mean(dim=1)  # (B, hidden_dim)

        # Score — higher score = more invariant violations
        violation = self.scorer(Z_pooled).squeeze(-1)  # (B,)

        return violation
