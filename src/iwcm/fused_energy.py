"""Fused IWCM — single projection, fused mean+max, fast std via variance formula."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class FusedIWCMEnergy(nn.Module):
    def __init__(self, d_slot, d_action=11, hidden=128, num_slots=8):
        super().__init__()
        self.hidden = hidden
        self.shared = nn.Linear(d_slot, hidden)
        self.head = nn.Sequential(
            nn.Linear(hidden * 3, hidden), nn.GELU(),
            nn.Linear(hidden, 3))
        self.register_buffer("lambdas", torch.tensor([1.0, 1.0, 1.5]))

    def forward(self, z0, A, Z):
        B, H, N, d = Z.shape
        Zf = self.shared(Z)
        Z_mean = Zf.mean(dim=1)
        Z_max = Zf.amax(dim=1)
        Z_sq = Zf * Zf
        Z_var = F.relu(Z_sq.mean(dim=1) - Z_mean * Z_mean)
        Z_std = torch.sqrt(Z_var + 1e-5)
        Zs = torch.cat([Z_mean, Z_max, Z_std], dim=-1)
        scores = self.head(Zs)
        agg = 0.3 * scores.mean(dim=1) + 0.7 * scores.amax(dim=1)
        return (agg * self.lambdas).sum(dim=-1)

    def score_acceptance(self, z0, A, Z):
        return torch.sigmoid(-self.forward(z0, A, Z))
