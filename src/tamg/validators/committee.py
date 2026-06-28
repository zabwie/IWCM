"""Validator Committee — 8 structurally asymmetric validators (Appendix C).

V1: Local transition likelihood (MLP, Δt=1)
V2: Forward-backward cycle consistency (bidirectional GRU)
V3: Slot correspondence consistency (contrastive, Δt=H/2)
V4: Invariant-subspace stability (slow-feature analysis)
V5: Action-locality consistency (causal attention)
V6: Temporal reachability score (learned classifier)
V7: Multi-view/augmentation consistency
V8: Repair difficulty under W_θ

No two validators share an encoder. Validators are frozen during
corruptor updates to prevent co-adaptation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Dict, Tuple
from src.utils.base import BaseModel


class Validator(BaseModel):
    """Base class for all validators in the committee."""
    name: str = "base"

    def forward(self, Z: torch.Tensor, z0: Optional[torch.Tensor] = None,
                A: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Score worldline validity. Higher = more likely valid."""
        raise NotImplementedError


class V1LocalTransition(Validator):
    """Local transition likelihood using MLP on adjacent slot pairs.

    2-layer MLP, horizon Δt=1, no action conditioning.
    """

    name = "v1_local"

    def __init__(self, d_state: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(d_state * 2, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, Z, z0=None, A=None):
        B, H, N, d = Z.shape
        if H < 2:
            return torch.zeros(B, device=Z.device)
        z_t = Z[:, :-1].reshape(-1, N, d)
        z_tp1 = Z[:, 1:].reshape(-1, N, d)
        pair = torch.cat([z_t, z_tp1], dim=-1)
        scores = torch.sigmoid(self.mlp(pair)).squeeze(-1)
        scores = scores.reshape(B, H - 1, N).mean(dim=-1).mean(dim=-1)
        return scores


class V2CycleConsistency(Validator):
    """Forward-backward cycle consistency using bidirectional GRU.

    Trained on forward-backward reconstruction.
    """

    name = "v2_cycle"

    def __init__(self, d_state: int):
        super().__init__()
        self.gru = nn.GRU(d_state, 128, bidirectional=True, batch_first=True)
        self.scorer = nn.Linear(256, 1)

    def forward(self, Z, z0=None, A=None):
        B, H, N, d = Z.shape
        Z_flat = Z.reshape(B, H * N, d)  # (B, H*N, d)
        _, h = self.gru(Z_flat)  # h: (2, B, 128)
        h_cat = torch.cat([h[0], h[1]], dim=-1)  # (B, 256)
        return torch.sigmoid(self.scorer(h_cat)).squeeze(-1)


class V3SlotCorrespondence(Validator):
    """Slot correspondence via contrastive matching over Δt=H/2."""

    name = "v3_correspondence"

    def __init__(self, d_state: int):
        super().__init__()
        self.proj = nn.Linear(d_state, 64)
        self.scorer = nn.Sequential(
            nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1),
        )

    def forward(self, Z, z0=None, A=None):
        B, H, N, d = Z.shape
        half = H // 2
        if half < 1:
            return torch.zeros(B, device=Z.device)
        early = self.proj(Z[:, :half].mean(dim=1))  # (B, N, 64)
        late = self.proj(Z[:, half:].mean(dim=1))   # (B, N, 64)
        sim = F.cosine_similarity(early, late, dim=-1).mean(dim=-1)  # (B,)
        return sim  # higher = more consistent


class V4InvariantStability(Validator):
    """Invariant-subspace stability — slow-feature analysis on content.

    Measures how stable the content subspace is over time.
    No action conditioning.
    """

    name = "v4_invariant"

    def __init__(self, d_state: int, content_dim: int = 32):
        super().__init__()
        self.content_dim = content_dim

    def forward(self, Z, z0=None, A=None):
        content = Z[..., :self.content_dim]
        diff = (content[:, 1:] - content[:, :-1]).pow(2).mean(dim=(-2, -1))
        stability = 1.0 / (1.0 + diff.mean(dim=-1))
        return stability


class V5ActionLocality(Validator):
    """Action-locality: causal attention estimator, action-conditioned.

    Learns which slots should change given an action.
    """

    name = "v5_locality"

    def __init__(self, d_state: int, d_action: int = 11):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_state, 4, batch_first=True)
        self.action_proj = nn.Linear(d_action, d_state)

    def forward(self, Z, z0=None, A=None):
        B, H, N, d = Z.shape
        if A is None or H < 2:
            return torch.zeros(B, device=Z.device)
        a_emb = self.action_proj(A.mean(dim=1)).unsqueeze(1).expand(-1, N, -1)  # (B, N, d)
        z_emb = Z.mean(dim=1)  # (B, N, d)
        attn_out, _ = self.attn(a_emb, z_emb, z_emb)
        return attn_out.norm(dim=(-2, -1)) / d  # (B,)


class V6Reachability(Validator):
    """Temporal reachability: classifier from z0 to z_t."""

    name = "v6_reachability"

    def __init__(self, d_state: int):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(d_state * 2, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

    def forward(self, Z, z0=None, A=None):
        B, H, N, d = Z.shape
        if z0 is None:
            return torch.ones(B, device=Z.device)
        z0_exp = z0.unsqueeze(1).expand(-1, H, -1, -1)
        pair = torch.cat([z0_exp, Z], dim=-1)
        reach = self.classifier(pair).squeeze(-1)
        return reach.mean(dim=(-2, -1))


class V7AugmentationConsistency(Validator):
    """Multi-view consistency: score invariance to augmentations.

    Adds Gaussian noise to Z and checks if score changes.
    """

    name = "v7_augmentation"

    def __init__(self, d_state: int):
        super().__init__()
        self.scorer = nn.Sequential(
            nn.Linear(d_state, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, Z, z0=None, A=None):
        # Score original
        s_orig = self.scorer(Z).squeeze(-1).mean(dim=(-2, -1))  # (B,)
        # Score augmented
        noise = torch.randn_like(Z) * 0.01
        s_aug = self.scorer(Z + noise).squeeze(-1).mean(dim=(-2, -1))
        # Consistency = 1 / (1 + |s_orig - s_aug|)
        consistency = 1.0 / (1.0 + (s_orig - s_aug).abs())
        return consistency


class V8RepairDifficulty(Validator):
    """Repair difficulty under W_θ.

    Measures how many refinement steps it takes the world model
    to reduce energy below a threshold. More steps = more likely invalid.
    (This is coupled to the IWCM energy function.)
    """

    name = "v8_repair"

    def __init__(self, d_state: int, max_steps: int = 20):
        super().__init__()
        self.max_steps = max_steps

    def set_energy_fn(self, energy_fn):
        self.energy_fn = energy_fn

    def forward(self, Z, z0=None, A=None):
        B = Z.shape[0]
        if not hasattr(self, "energy_fn") or z0 is None or A is None:
            return torch.zeros(B, device=Z.device)
        # Rough estimate: check initial energy
        with torch.no_grad():
            energy = self.energy_fn(z0, A, Z)
        # Higher energy = harder to repair = lower validity score
        validity = torch.exp(-energy / self.max_steps)
        return validity.clamp(0.0, 1.0)


# ═══════════════════════════════════════════════════════════
# Validator Committee
# ═══════════════════════════════════════════════════════════

class ValidatorCommittee(BaseModel):
    """Committee of 8 structurally asymmetric validators.

    Causal invalidity is identified by structured disagreement among
    independent consistency tests — not by any single arbiter.
    """

    def __init__(
        self, d_state: int, d_action: int = 11, content_dim: int = 32,
    ):
        super().__init__()
        self.validators: List[Validator] = [
            V1LocalTransition(d_state),
            V2CycleConsistency(d_state),
            V3SlotCorrespondence(d_state),
            V4InvariantStability(d_state, content_dim),
            V5ActionLocality(d_state, d_action),
            V6Reachability(d_state),
            V7AugmentationConsistency(d_state),
            V8RepairDifficulty(d_state),
        ]
        self.validator_names = [v.name for v in self.validators]

    def forward(
        self, Z: torch.Tensor, z0: torch.Tensor, A: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Evaluate all validators.

        Returns:
            Dict mapping validator name to validity score (B,).
        """
        return {
            v.name: v(Z, z0, A) for v in self.validators
        }

    def get_local_scores(
        self, Z: torch.Tensor, z0: torch.Tensor, A: torch.Tensor,
    ) -> torch.Tensor:
        """Get scores from local validators (V1, V7)."""
        v1 = self.validators[0](Z, z0, A)
        v7 = self.validators[6](Z, z0, A)
        return torch.stack([v1, v7], dim=-1)  # (B, 2)

    def get_global_scores(
        self, Z: torch.Tensor, z0: torch.Tensor, A: torch.Tensor,
    ) -> torch.Tensor:
        """Get scores from global validators (V2, V3, V4, V6)."""
        v2 = self.validators[1](Z, z0, A)
        v3 = self.validators[2](Z, z0, A)
        v4 = self.validators[3](Z, z0, A)
        v6 = self.validators[5](Z, z0, A)
        return torch.stack([v2, v3, v4, v6], dim=-1)  # (B, 4)

    def freeze_for_corruptor_update(self) -> None:
        """Freeze validators during corruptor training."""
        for p in self.parameters():
            p.requires_grad = False

    def unfreeze(self) -> None:
        """Unfreeze validators."""
        for p in self.parameters():
            p.requires_grad = True
