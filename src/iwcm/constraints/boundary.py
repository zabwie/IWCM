"""Boundary constraint head — anchors all future states to real z0.

Implements C_boundary(z0, Z, A) = Σ_{t=1}^H d(zt, R(z0, A_{0:t}))
where R computes reachability from z0 given actions.

Cross-attention from z0 to all z_t, with learned reachability scoring.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .base import ConstraintHead


class BoundaryHead(ConstraintHead):
    """Boundary constraint: every z_t must be reachable from z0.

    Uses cross-attention from a learned z0 embedding to each z_t,
    producing a reachability score that penalizes drift.
    """

    def __init__(
        self,
        d_state: int,
        d_action: int = 11,
        hidden_dim: int = 256,
        num_layers: int = 2,
    ):
        super().__init__(d_state, d_action, hidden_dim)

        # z0 encoder: project initial state to query space
        self.z0_proj = nn.Linear(d_state, hidden_dim)

        # Cross-attention: z0 attends to target state
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=4,
            batch_first=False,  # (seq, batch, dim)
        )

        # Reachability scorer
        layers = []
        in_dim = hidden_dim * 2  # z0_query + zt_value
        for i in range(num_layers):
            out_dim = hidden_dim if i < num_layers - 1 else 1
            layers.append(nn.Linear(in_dim, out_dim))
            if i < num_layers - 1:
                layers.append(nn.ReLU())
            in_dim = hidden_dim
        self.reach_scorer = nn.Sequential(*layers)

    def forward(
        self, z0: torch.Tensor, A: torch.Tensor, Z: torch.Tensor
    ) -> torch.Tensor:
        B, H, d = Z.shape

        # Encode z0 as query
        z0_query = self.z0_proj(z0)  # (B, hidden_dim)

        # Cross-attend from z0 to all z_t
        # Reshape: z0 as query (1, B, hidden), Z as key/value (H, B, hidden)
        Z_hidden = Z.unsqueeze(-1).expand(-1, -1, -1, self.hidden_dim // d * d)
        # Simpler: just project each z_t
        Z_proj = nn.Linear(d, self.hidden_dim, device=Z.device)(Z)  # (B, H, hidden)

        # Cross attention
        query = z0_query.unsqueeze(0)  # (1, B, hidden)
        key_val = Z_proj.transpose(0, 1)  # (H, B, hidden)

        attn_output, _ = self.cross_attn(query, key_val, key_val)
        attn_output = attn_output.squeeze(0)  # (B, hidden)

        # Concatenate z0 query with attention output
        combined = torch.cat([z0_query, attn_output], dim=-1)  # (B, 2*hidden)

        # Score reachability — lower is better (penalty form)
        reach_scores = self.reach_scorer(combined).squeeze(-1)  # (B,)

        # Also penalize per-timestep distance from z0
        z0_expanded = z0.unsqueeze(1).expand(-1, H, -1)  # (B, H, d)
        per_step_dist = F.mse_loss(Z, z0_expanded, reduction="none").mean(dim=-1)  # (B, H)
        mean_drift = per_step_dist.mean(dim=-1)  # (B,)

        return reach_scores + 0.1 * mean_drift
