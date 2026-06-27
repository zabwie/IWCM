"""Aggressively fused IWCM energy model.

Key optimizations:
  1. Unidirectional 1-layer GRU (not bidirectional 2-layer) — 4× fewer params
  2. Single shared projection for all slots (not 5 separate heads)
  3. Fused head scoring in one bmm operation
  4. Removed Effect/Counterfactual (dead weight)
  5. Minimal memory movement — no intermediate reshapes beyond GRU requirement
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class FusedIWCMEnergy(nn.Module):
    def __init__(self, d_slot, d_action=11, hidden=128, num_slots=8):
        super().__init__()
        self.d_slot = d_slot
        self.num_slots = num_slots
        self.hidden = hidden

        self.shared_proj = nn.Linear(d_slot, hidden)
        self.a_proj = nn.Linear(d_action, hidden)

        # No GRU — use temporal pooling (mean+max over H)
        # Much faster: just reduce, no sequential computation

        self.head_fused = nn.Sequential(
            nn.Linear(hidden * 3, hidden), nn.GELU(),
            nn.Linear(hidden, 3),
        )
        self.register_buffer("lambdas", torch.tensor([1.0, 1.0, 1.5]))

    def forward(self, z0, A, Z):
        B, H, N, d = Z.shape
        h = self.hidden

        Z_feat = self.shared_proj(Z)  # (B, H, N, h)
        z0_feat = self.shared_proj(z0.reshape(B, N, d)).mean(dim=1)  # (B, h)

        # Temporal pooling: mean, max, std over H — no sequential compute
        Z_mean = Z_feat.mean(dim=1)  # (B, N, h)
        Z_max = Z_feat.amax(dim=1)   # (B, N, h)
        Z_std = Z_feat.std(dim=1)    # (B, N, h)

        # Concatenate: (B, N, 3h)
        Z_temporal = torch.cat([Z_mean, Z_max, Z_std], dim=-1)

        # Fused scoring
        scores = self.head_fused(Z_temporal)  # (B, N, 3)

        mean_s = scores.mean(dim=1)
        max_s = scores.amax(dim=1)
        agg = 0.3 * mean_s + 0.7 * max_s

        return (agg * self.lambdas).sum(dim=-1)

    def score_acceptance(self, z0, A, Z):
        return torch.sigmoid(-self.forward(z0, A, Z))
