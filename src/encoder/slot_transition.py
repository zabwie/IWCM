"""Slot transition predictor — learned next-frame slot initialization.

SAVi-style transition model: given slots at time t and action a_t,
predicts slot initialization for time t+1. This provides slot attention
with a strong prior, dramatically reducing slot permutation across frames.

Architecture:
  s_{t+1}^init = s_t + MLP([s_t, a_t])

with optional GRU state for temporal reasoning.
"""

import torch
import torch.nn as nn
from typing import Optional


class SlotTransitionPredictor(nn.Module):
    """Learned next-frame slot initialization from current slots and action.

    Takes per-slot features at time t and the action a_t, predicts
    the initialization for slot attention at time t+1.

    Args:
        slot_dim: Dimension per slot (d).
        d_action: Action encoding dimension.
        hidden_dim: Hidden dimension for the transition MLP.
        use_gru: If True, use GRU for temporal state tracking.
        dropout: Dropout rate (0.0 = no dropout).
    """

    def __init__(
        self,
        slot_dim: int = 64,
        d_action: int = 11,
        hidden_dim: int = 256,
        use_gru: bool = False,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.slot_dim = slot_dim
        self.d_action = d_action
        self.use_gru = use_gru

        # Action projection to match slot dimension
        self.action_proj = nn.Linear(d_action, slot_dim)

        # Transition MLP: [slot, action_emb, slot+action_emb] → delta
        # Triple input: raw slot, action embedding, and interaction term
        input_dim = slot_dim * 2 + slot_dim  # slot + action_proj + (slot * action_proj)
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, slot_dim),
        )

        # Optional GRU for temporal state
        if use_gru:
            self.gru = nn.GRUCell(slot_dim, slot_dim)
            self.gru_hidden: Optional[torch.Tensor] = None

        # Learnable initial hidden state for slot attention bootstrapping
        # Broadcast to (B, N, slot_dim) for the first frame
        self.register_buffer(
            "default_init",
            torch.zeros(1, 1, slot_dim),
        )

    def _interaction(self, slot: torch.Tensor, action_emb: torch.Tensor) -> torch.Tensor:
        """Compute multiplicative interaction: slot ⊙ action_emb."""
        return slot * action_emb

    def forward(
        self,
        slots_t: torch.Tensor,
        action_t: torch.Tensor,
    ) -> torch.Tensor:
        """Predict slot initialization for time t+1.

        Args:
            slots_t: Current slots at time t, shape (B, N, slot_dim).
            action_t: Action at time t, shape (B, d_action).

        Returns:
            Predicted slots for time t+1, shape (B, N, slot_dim).
        """
        B, N, d = slots_t.shape
        assert action_t.shape == (B, self.d_action), (
            f"Expected action shape (B={B}, d_action={self.d_action}), got {action_t.shape}"
        )

        # Expand action to per-slot: (B, d_action) → (B, N, d_action)
        a_expanded = action_t.unsqueeze(1).expand(B, N, self.d_action)

        # Project action to slot space
        a_emb = self.action_proj(a_expanded)  # (B, N, slot_dim)

        # Multiplicative interaction
        interaction = self._interaction(slots_t, a_emb)  # (B, N, slot_dim)

        # Concatenate: [slot, action_emb, interaction]
        mlp_input = torch.cat([slots_t, a_emb, interaction], dim=-1)  # (B, N, 3*d)

        # Predict delta
        delta = self.mlp(mlp_input)  # (B, N, slot_dim)

        # Residual: default behavior is "stay the same"
        next_slots = slots_t + delta

        # Optional GRU refinement
        if self.use_gru:
            flat = next_slots.reshape(B * N, d)
            if self.gru_hidden is None or self.gru_hidden.shape[0] != B * N:
                self.gru_hidden = torch.zeros(B * N, d, device=slots_t.device)
            self.gru_hidden = self.gru(flat, self.gru_hidden)
            next_slots = self.gru_hidden.reshape(B, N, d)

        return next_slots

    def reset_state(self):
        """Reset GRU hidden state (call at start of each sequence)."""
        self.gru_hidden = None

    def predict_sequence(
        self,
        slots_0: torch.Tensor,
        actions: torch.Tensor,
    ) -> torch.Tensor:
        """Predict slot sequence from initial slots and action sequence.

        Args:
            slots_0: Initial slots at time 0, shape (B, N, slot_dim).
            actions: Action sequence, shape (B, H, d_action).

        Returns:
            Predicted slot sequence, shape (B, H, N, slot_dim).
            Note: slots_0 is NOT included; output is predictions for t=1..H.
        """
        B, N, d = slots_0.shape
        H = actions.shape[1]
        self.reset_state()

        preds = []
        slots = slots_0
        for t in range(H):
            slots = self.forward(slots, actions[:, t])
            preds.append(slots)

        return torch.stack(preds, dim=1)  # (B, H, N, slot_dim)


def make_identity_init(
    num_slots: int,
    slot_dim: int,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    """Create orthogonal initial slot vectors for bootstrapping.

    Uses a random orthogonal matrix to ensure slots start maximally
    distinct, reducing initial permutation ambiguity.

    Args:
        num_slots: Number of slots (N).
        slot_dim: Slot dimension (d).
        batch_size: Batch size (B).
        device: Target device.

    Returns:
        Initial slots of shape (B, N, slot_dim).
    """
    # Generate random orthogonal vectors
    q, _ = torch.linalg.qr(torch.randn(slot_dim, num_slots, device=device))
    init = q.t()[:num_slots]  # (N, slot_dim)
    return init.unsqueeze(0).expand(batch_size, num_slots, slot_dim) * 0.1
