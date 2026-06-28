"""Optimized slot-aware constraint heads — shared backbone, parallel temporal ops."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class FastSlotIWCMEnergy(nn.Module):
    """Slot-aware IWCM with shared encoder + small per-head detectors.

    Key optimizations:
      1. Single shared projection (not 5 separate projections)
      2. Parallel temporal 1D conv (not sequential bidirectional GRU)
      3. Fused head computation in one forward pass
      4. Removed dead-weight Effect/Counterfactual complexity
    """

    def __init__(self, d_slot, d_action=11, hidden=96, num_slots=8):
        super().__init__()
        self.d_slot = d_slot
        self.num_slots = num_slots
        self.hidden = hidden

        # === Single shared backbone ===
        self.z_proj = nn.Linear(d_slot, hidden)
        self.a_proj = nn.Linear(d_action, hidden)

        # === Parallel temporal encoder (Conv1D over H) ===
        # Minimal 2-layer depthwise+pointwise for speed
        self.temporal = nn.Sequential(
            nn.Conv1d(hidden, hidden, 3, padding=1, groups=hidden),
            nn.GELU(),
            nn.Conv1d(hidden, hidden, 1),
        )

        # === Tiny per-head scorers (shared features, separate detectors) ===
        self.boundary_scorer = nn.Sequential(nn.Linear(hidden, hidden // 2), nn.GELU(), nn.Linear(hidden // 2, 1))
        self.local_scorer = nn.Sequential(nn.Linear(hidden, hidden // 2), nn.GELU(), nn.Linear(hidden // 2, 1))
        self.invariant_scorer = nn.Sequential(nn.Linear(hidden, hidden // 2), nn.GELU(), nn.Linear(hidden // 2, 1))
        self.effect_scorer = nn.Sequential(nn.Linear(hidden, hidden // 2), nn.GELU(), nn.Linear(hidden // 2, 1))
        self.cf_scorer = nn.Sequential(nn.Linear(hidden * 2, hidden // 2), nn.GELU(), nn.Linear(hidden // 2, 1))

        # Lambda weights
        self.lambdas = {"boundary": 1.0, "local": 1.0, "invariant": 1.5, "effect": 1.0, "counterfactual": 0.5}

    def forward(self, z0, A, Z):
        B, H, N, d = Z.shape
        h = self.hidden

        # === Shared projection (done ONCE) ===
        Z_feat = self.z_proj(Z)  # (B, H, N, h)
        z0_feat = self.z_proj(z0.reshape(B, N, d)).unsqueeze(1)  # (B, 1, N, h)
        A_feat = self.a_proj(A)  # (B, H, h)

        # === Temporal processing (parallel Conv1D over H) ===
        # Reshape: (B, H, N, h) → (B*N, h, H) for Conv1d
        Z_temporal = Z_feat.permute(0, 2, 3, 1).reshape(B * N, h, H)  # (B*N, h, H)
        Z_encoded = self.temporal(Z_temporal)  # (B*N, h, H)
        Z_encoded = Z_encoded.reshape(B, N, h, H).permute(0, 3, 1, 2)  # (B, H, N, h)

        # === Head-specific scoring (tiny MLPs on shared features) ===
        # Boundary: compare each Z_t with z0
        boundary = (Z_encoded * z0_feat).mean(dim=-1)  # (B, H, N) — dot product attention
        b_score = self.boundary_scorer(Z_encoded.mean(dim=1).mean(dim=1)).squeeze(-1)  # (B,)

        # Local: compare adjacent pairs
        if H > 1:
            l_pair = torch.cat([Z_encoded[:, :-1, :, :h//2], Z_encoded[:, 1:, :, :h//2]], dim=-1)
            local = self.local_scorer(l_pair.mean(dim=1).mean(dim=1)).squeeze(-1)  # (B,)
        else:
            local = torch.zeros(B, device=Z.device)

        # Invariant: temporal stability
        inv = self.invariant_scorer(Z_encoded.mean(dim=1).mean(dim=1)).squeeze(-1)  # (B,)

        # Effect: action-conditioned change detection
        if H > 1:
            a_exp = A_feat.unsqueeze(2).expand(-1, -1, N, -1)  # (B, H, N, h)
            effect_input = (Z_encoded * a_exp).mean(dim=1).mean(dim=1)  # (B, h)
            effect = self.effect_scorer(effect_input).squeeze(-1)
        else:
            effect = torch.zeros(B, device=Z.device)

        # Counterfactual: early vs late consistency
        if H > 1:
            early = Z_encoded[:, :H//2].mean(dim=1)  # (B, N, h)
            late = Z_encoded[:, H//2:].mean(dim=1)   # (B, N, h)
            cf_input = torch.cat([early.mean(dim=1), late.mean(dim=1)], dim=-1)  # (B, 2h)
            cf = self.cf_scorer(cf_input).squeeze(-1)
        else:
            cf = torch.zeros(B, device=Z.device)

        # === Aggregate ===
        energy = (self.lambdas["boundary"] * b_score +
                  self.lambdas["local"] * local +
                  self.lambdas["invariant"] * inv +
                  self.lambdas["effect"] * effect +
                  self.lambdas["counterfactual"] * cf)
        return energy

    def score_acceptance(self, z0, A, Z):
        return torch.sigmoid(-self.forward(z0, A, Z))
