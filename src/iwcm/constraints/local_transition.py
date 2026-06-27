"""Local transition constraint head.

Implements C_local(Z, A) = Σ_{t=0}^{H-1} c_loc(z_t, z_{t+1}, a_t).

Adjacent states must be locally plausible. Critically, z_{t+1} is a
free variable SCORED AGAINST z_t, not produced from it — this is what
distinguishes IWCM from autoregressive models.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .base import ConstraintHead


class LocalTransitionHead(ConstraintHead):
    """Scores local plausibility of adjacent state pairs with action conditioning.

    Uses self-attention on (z_t, z_{t+1}) pairs with action embeddings,
    then scores the plausibility of each transition.
    """

    def __init__(
        self,
        d_state: int,
        d_action: int = 11,
        hidden_dim: int = 256,
    ):
        super().__init__(d_state, d_action, hidden_dim)

        # Joint encoding of (z_t, a_t) → hidden
        self.transition_proj = nn.Linear(d_state + d_action, hidden_dim)

        # Self-attention over pairs
        self.self_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=4,
            batch_first=True,  # (B, seq, dim)
        )

        # Plausibility scorer
        self.scorer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(
        self, z0: torch.Tensor, A: torch.Tensor, Z: torch.Tensor
    ) -> torch.Tensor:
        B, H, d = Z.shape

        if H < 2:
            return torch.zeros(B, device=Z.device)

        # Build (z_t, a_t) pairs for t=0..H-1
        z_t = Z[:, :-1, :]  # (B, H-1, d)
        z_tp1 = Z[:, 1:, :]  # (B, H-1, d)
        a_t = A[:, :-1, :] if A.shape[1] == H else A  # (B, H-1, d_action)

        # Joint encoding: concat z_t, a_t → hidden
        joint = torch.cat([z_t, a_t], dim=-1)  # (B, H-1, d + d_action)
        hidden = self.transition_proj(joint)  # (B, H-1, hidden_dim)

        # Self-attention over all transitions in the worldline
        attended, _ = self.self_attn(hidden, hidden, hidden)  # (B, H-1, hidden_dim)

        # Score each transition's plausibility
        scores = self.scorer(attended).squeeze(-1)  # (B, H-1)

        # Higher score = more plausible, so violation = 1 - mean plausibility
        violation = 1.0 - torch.sigmoid(scores).mean(dim=-1)  # (B,)

        return violation
