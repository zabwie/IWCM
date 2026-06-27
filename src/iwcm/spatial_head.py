"""Spatial constraint head — detects teleport, illegal displacement, and spatial anomalies.

Operates on raw position (ch 5:6) and velocity (ch 7:8) channels from the oracle
slot encoder. Unlike the main 3-head scorer which operates on shared(Z) — a nonlinear
mix of all 19 channels — this head directly reads the physical position/velocity
features. This prevents the signal from being diluted by type, identity hash, and
other channels.

Architecture:
  raw_vel  = Z[:, :, :, 7:9]              # velocity (B, H, N, 2)
  vel_feat = spatial_proj(raw_vel)         # Linear(2, hidden_s) → (B, H, N, hidden_s)
  pooled   = vel_feat.amax(dim=1).amax(1)  # max over H then N → (B, hidden_s)
  score    = spatial_scorer(pooled)        # Linear(hidden_s, 1) → (B,)

Max pooling over H and N ensures ANY single anomalous timestep for ANY object
triggers detection — critical for teleport (one bad step → violation).
"""
import torch
import torch.nn as nn


class SpatialConstraintHead(nn.Module):
    """Detects spatial violations (teleport, illegal displacement) from raw velocity."""

    def __init__(self, d_slot: int = 19, hidden_s: int = 8):
        super().__init__()
        self.hidden_s = hidden_s

        # Velocity channels are at indices 7:9 in the oracle slot encoder
        self.vel_start = 7
        self.vel_end = 9

        # Tiny projection: 2 velocity dims → hidden_s spatial features
        self.spatial_proj = nn.Linear(2, hidden_s)

        # Scorer: hidden_s → 1
        self.spatial_scorer = nn.Sequential(
            nn.Linear(hidden_s, hidden_s),
            nn.ReLU(),
            nn.Linear(hidden_s, 1),
        )

    def forward(self, Z: torch.Tensor) -> torch.Tensor:
        """Compute spatial violation score.

        Args:
            Z: (B, H, N, d_slot) — full slot encoding per timestep.

        Returns:
            score: (B,) — spatial violation score per batch element.
        """
        # Extract raw velocity channels: (B, H, N, 2)
        raw_vel = Z[:, :, :, self.vel_start:self.vel_end]

        # Project to spatial feature space: (B, H, N, hidden_s)
        vel_feat = self.spatial_proj(raw_vel)

        # Max pool over H (any timestep) then N (any object)
        # This catches single-timestep anomalies like teleport
        pooled = vel_feat.amax(dim=1).amax(dim=1)  # (B, hidden_s)

        # Score
        return self.spatial_scorer(pooled).squeeze(-1)  # (B,)


class DisplacementHead(nn.Module):
    """Specialized teleport detector using raw displacement magnitude.

    Even simpler: directly compute max(||velocity||) and pass through
    a learnable logistic function. This is the hand-crafted version that
    achieved 100% teleport detection with 0% false positives.

    score = sigmoid(scale * (max_displacement - threshold))
    where scale and threshold are learned parameters.
    """

    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(10.0))
        self.threshold = nn.Parameter(torch.tensor(0.13))

    def forward(self, Z: torch.Tensor) -> torch.Tensor:
        raw_vel = Z[:, :, :, 7:9]
        mag = raw_vel.pow(2).sum(dim=-1).sqrt()  # (B, H, N)
        max_disp = mag.amax(dim=1).amax(dim=1)    # (B,)
        return torch.sigmoid(self.scale * (max_disp - self.threshold))


# ---------------------------------------------------------------------------
# Full energy model with spatial head
# ---------------------------------------------------------------------------

class SpatialIWCMEnergy(nn.Module):
    """IWCM energy with spatial constraint heads for teleport and deletion detection.

    Extends FusedIWCMEnergy with heads that operate on raw encoder channels:
      - Velocity head: detects teleport via max(||velocity||)
      - Existence head: detects delete via slot occupancy changes

    Energy = core_3head + lambda_vel * vel_score + lambda_exist * exist_score

    Args:
        d_slot, d_action, hidden, num_slots: Same as FusedIWCMEnergy.
        hidden_s: Hidden dim for spatial MLP head.
        lambda_vel: Weight for velocity/teleport head (default 1.0).
        lambda_exist: Weight for existence/deletion head (default 1.5).
    """

    def __init__(self, d_slot: int, d_action: int = 11, hidden: int = 128,
                 num_slots: int = 8, hidden_s: int = 8,
                 lambda_vel: float = 1.0, lambda_exist: float = 1.5):
        super().__init__()
        self.hidden = hidden
        self.lambda_vel = lambda_vel
        self.lambda_exist = lambda_exist

        # Core 3-head scorer
        self.shared = nn.Linear(d_slot, hidden)
        self.head = nn.Sequential(
            nn.Linear(hidden * 3, hidden),
            nn.GELU(),
            nn.Linear(hidden, 3),
        )
        self.register_buffer("lambdas", torch.tensor([1.0, 1.0, 1.5]))

        # Spatial heads
        self.spatial = SpatialConstraintHead(d_slot, hidden_s)
        self.existence = ExistenceHead()

    def forward(self, z0, A, Z):
        B, H, N_in, d = Z.shape

        # Core 3-head scorer
        Zf = self.shared(Z)
        Z_mean = Zf.mean(dim=1)
        Z_max = Zf.amax(dim=1)
        Z_sq = Zf * Zf
        Z_var = torch.nn.functional.relu(Z_sq.mean(dim=1) - Z_mean * Z_mean)
        Z_std = torch.sqrt(Z_var + 1e-5)
        Zs = torch.cat([Z_mean, Z_max, Z_std], dim=-1)
        scores = self.head(Zs)
        agg = 0.3 * scores.mean(dim=1) + 0.7 * scores.amax(dim=1)
        core_energy = (agg * self.lambdas).sum(dim=-1)

        # Spatial heads
        vel_score = self.spatial(Z)
        exist_score = self.existence(Z)

        return core_energy + self.lambda_vel * vel_score + self.lambda_exist * exist_score

    def score_acceptance(self, z0, A, Z):
        return torch.sigmoid(-self.forward(z0, A, Z))

    def per_head(self, z0, A, Z):
        B, H, N_in, d = Z.shape
        Zf = self.shared(Z)
        Z_mean = Zf.mean(dim=1)
        Z_max = Zf.amax(dim=1)
        Z_sq = Zf * Zf
        Z_var = torch.nn.functional.relu(Z_sq.mean(dim=1) - Z_mean * Z_mean)
        Z_std = torch.sqrt(Z_var + 1e-5)
        Zs = torch.cat([Z_mean, Z_max, Z_std], dim=-1)
        scores = self.head(Zs)
        agg = 0.3 * scores.mean(dim=1) + 0.7 * scores.amax(dim=1)
        return {
            "boundary": agg[:, 0],
            "local": agg[:, 1],
            "invariant": agg[:, 2],
            "velocity": self.spatial(Z),
            "existence": self.existence(Z),
        }


class ExistenceHead(nn.Module):
    """Detects object deletion via slot occupancy changes.

    Empty slots are all-zeros in the oracle encoder. A deletion makes
    an occupied slot become all-zeros. This head detects the maximum
    drop in slot occupancy across timesteps.

    occupancy = sum(|Z[:,:,:,0:5]|, dim=-1)  # type channels → (B,H,N)
    max_drop  = max(occupancy[t] - occupancy[t+1]) over t
    score     = sigmoid(scale * (max_drop - threshold))
    """

    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(5.0))
        self.threshold = nn.Parameter(torch.tensor(0.5))

    def forward(self, Z: torch.Tensor) -> torch.Tensor:
        # Type channels: 0-4 (agent, key, door, box, occluder one-hots)
        # Sum of type channels = 1.0 for occupied slots, 0.0 for empty
        occupancy = Z[:, :, :, 0:5].abs().sum(dim=-1)  # (B, H, N)

        # Max drop in occupancy: positive when a slot goes from occupied→empty
        occupancy_drop = occupancy[:, :-1] - occupancy[:, 1:]  # (B, H-1, N)
        max_drop = occupancy_drop.amax(dim=1).amax(dim=1)  # (B,)

        return torch.sigmoid(self.scale * (max_drop - self.threshold))
