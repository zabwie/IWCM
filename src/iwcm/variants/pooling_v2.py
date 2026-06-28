"""Pooling IWCM v2 — delta features + law-specific heads + statistical temporal pooling.

Key additions over v1:
  1. Delta pooling: ΔZ = Z[:,1:] - Z[:,:-1] → capture discontinuity explicitly
  2. First/last/min features for richer temporal statistics
  3. Law-specific heads (conservation, identity, teleport, validity)
  4. All pooling is O(1) over H — no recurrence, no attention
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def fast_std(x, dim, eps=1e-5):
    """Faster std: sqrt(mean(x²) - mean(x)² + eps)."""
    m = x.mean(dim=dim)
    m2 = (x * x).mean(dim=dim)
    return torch.sqrt(F.relu(m2 - m * m) + eps)


class PoolingIWCMv2(nn.Module):
    def __init__(self, d_slot, d_action=11, hidden=96, num_slots=8):
        super().__init__()
        self.hidden = hidden

        # === Single shared projection ===
        self.shared = nn.Linear(d_slot, hidden)

        # === Statistical features dimension ===
        # Per-object: mean(h), max(h), min(h), std(h), first(h), last(h), last-first(h)
        # Delta: mean(|dh|), max(|dh|), std(dh)
        # Action: mean(A) over time
        # Total: 10 statistical features × hidden
        self.stat_dim = 10

        # === Fused statistical head: 10*hidden → 4 law scores ===
        self.head = nn.Sequential(
            nn.Linear(self.stat_dim * hidden, hidden * 2),
            nn.GELU(),
            nn.Linear(hidden * 2, hidden),
            nn.GELU(),
            nn.Linear(hidden, 4),  # conservation, identity, teleport, validity
        )

        self.register_buffer("lambdas", torch.tensor([1.5, 1.0, 1.2, 0.8]))

    def forward(self, z0, A, Z):
        B, H, N, d = Z.shape
        h = self.hidden

        # === Shared projection ===
        Z_feat = self.shared(Z)  # (B, H, N, h)

        # === Standard temporal statistics ===
        Z_mean = Z_feat.mean(dim=1)        # (B, N, h)
        Z_max = Z_feat.amax(dim=1)         # (B, N, h)
        Z_min = Z_feat.amin(dim=1)         # (B, N, h)
        Z_std = fast_std(Z_feat, dim=1)    # (B, N, h)
        Z_first = Z_feat[:, 0]             # (B, N, h)
        Z_last = Z_feat[:, -1]             # (B, N, h)
        Z_diff = Z_last - Z_first          # (B, N, h)

        # === Delta features (capture discontinuity) ===
        if H > 1:
            dZ = Z_feat[:, 1:] - Z_feat[:, :-1]  # (B, H-1, N, h)
            dZ_abs = dZ.abs()
            D_mean = dZ_abs.mean(dim=1)    # (B, N, h)
            D_max = dZ_abs.amax(dim=1)     # (B, N, h)
            D_std = fast_std(dZ, dim=1)    # (B, N, h)
        else:
            D_mean = torch.zeros(B, N, h, device=Z.device)
            D_max = torch.zeros(B, N, h, device=Z.device)
            D_std = torch.zeros(B, N, h, device=Z.device)

        # === Concatenate all statistics: (B, N, 10*h) ===
        stats = torch.cat([
            Z_mean, Z_max, Z_min, Z_std,
            Z_first, Z_last, Z_diff,
            D_mean, D_max, D_std,
        ], dim=-1)  # (B, N, 10h)

        # === Aggregate over objects: mean + max ===
        stat_mean = stats.mean(dim=1)  # (B, 10h)
        stat_max = stats.amax(dim=1)   # (B, 10h)
        agg = 0.4 * stat_mean + 0.6 * stat_max  # (B, 10h)

        # === Law-specific heads ===
        scores = self.head(agg)  # (B, 4)
        return (scores * self.lambdas).sum(dim=-1)  # (B,)

    def score_acceptance(self, z0, A, Z):
        return torch.sigmoid(-self.forward(z0, A, Z))
