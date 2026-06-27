"""Slot-aware constraint heads — preserve (B, H, N, d_slot) structure throughout."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class SlotConstraintHead(nn.Module):
    def __init__(self, d_slot, d_action=11, hidden_dim=128, num_slots=8):
        super().__init__()
        self.d_slot = d_slot; self.d_action = d_action
        self.hidden_dim = hidden_dim; self.num_slots = num_slots

    def aggregate(self, per_slot, alpha=1.0):
        """Pure max pooling — sparse violations need max, not mean.
        alpha=1.0 = max-only, alpha=0.5 = hybrid (previous default)."""
        return per_slot.amax(dim=(-2, -1)) if per_slot.dim() == 3 else per_slot.amax(dim=-1)


class SlotBoundaryHead(SlotConstraintHead):
    name = "boundary"
    def __init__(self, d_slot, d_action=11, hidden_dim=128, num_slots=8):
        super().__init__(d_slot, d_action, hidden_dim, num_slots)
        self.proj = nn.Linear(d_slot, hidden_dim)
        self.scorer = nn.Sequential(nn.Linear(hidden_dim * 2, hidden_dim//2), nn.ReLU(), nn.Linear(hidden_dim//2, 1))

    def forward(self, z0, A, Z):
        B, H, N, d = Z.shape
        z0_s = z0.reshape(B, N, d)
        Zp = self.proj(Z)
        z0p = self.proj(z0_s).unsqueeze(1).expand(-1, H, -1, -1)
        pair = torch.cat([Zp, z0p], dim=-1)
        per_slot = self.scorer(pair).squeeze(-1)
        return self.aggregate(per_slot)


class SlotLocalHead(SlotConstraintHead):
    name = "local"
    def __init__(self, d_slot, d_action=11, hidden_dim=128, num_slots=8):
        super().__init__(d_slot, d_action, hidden_dim, num_slots)
        self.mlp = nn.Sequential(nn.Linear(d_slot * 2 + d_action, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1))

    def forward(self, z0, A, Z):
        B, H, N, d = Z.shape
        if H < 2: return torch.zeros(B, device=Z.device)
        z_t = Z[:, :-1]; z_tp1 = Z[:, 1:]
        a_exp = A[:, :-1].unsqueeze(2).expand(-1, -1, N, -1)
        pair = torch.cat([z_t, a_exp, z_tp1], dim=-1)
        per_slot = self.mlp(pair).squeeze(-1)
        return self.aggregate(per_slot)


class SlotInvariantHead(SlotConstraintHead):
    name = "invariant"
    def __init__(self, d_slot, d_action=11, hidden_dim=128, num_slots=8, num_layers=2):
        super().__init__(d_slot, d_action, hidden_dim, num_slots)
        self.proj = nn.Linear(d_slot, hidden_dim)
        self.per_object_temporal = nn.GRU(hidden_dim, hidden_dim, num_layers=num_layers,
                                           batch_first=True, bidirectional=True)
        self.scorer = nn.Sequential(nn.Linear(hidden_dim * 2, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1))

    def forward(self, z0, A, Z):
        B, H, N, d = Z.shape
        Zp = self.proj(Z)  # (B, H, N, h)
        Zp = Zp.permute(0, 2, 1, 3).reshape(B * N, H, self.hidden_dim)  # (B*N, H, h)
        _, h_n = self.per_object_temporal(Zp)  # h_n: (2*layers, B*N, h)
        h_fwd = h_n[-2]; h_bwd = h_n[-1]  # (B*N, h)
        h_cat = torch.cat([h_fwd, h_bwd], dim=-1)  # (B*N, 2h)
        per_slot = self.scorer(h_cat).squeeze(-1).reshape(B, N)  # (B, N)
        return self.aggregate(per_slot)


class SlotEffectHead(SlotConstraintHead):
    name = "effect"
    def __init__(self, d_slot, d_action=11, hidden_dim=128, num_slots=8):
        super().__init__(d_slot, d_action, hidden_dim, num_slots)
        self.scope = nn.Sequential(nn.Linear(d_slot + d_action, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1), nn.Sigmoid())

    def forward(self, z0, A, Z):
        B, H, N, d = Z.shape
        if H < 2: return torch.zeros(B, device=Z.device)
        z_t = Z[:, :-1]; delta = Z[:, 1:] - Z[:, :-1]
        a_exp = A[:, :-1].unsqueeze(2).expand(-1, -1, N, -1)
        s = self.scope(torch.cat([z_t, a_exp], dim=-1)).squeeze(-1)
        per_slot = (1 - s) * delta.pow(2).mean(dim=-1)
        return self.aggregate(per_slot)


class SlotCounterfactualHead(SlotConstraintHead):
    name = "counterfactual"
    def __init__(self, d_slot, d_action=11, hidden_dim=128, num_slots=8):
        super().__init__(d_slot, d_action, hidden_dim, num_slots)
        self.proj = nn.Linear(d_slot, hidden_dim)
        self.comp = nn.Sequential(nn.Linear(hidden_dim * 2, hidden_dim//2), nn.ReLU(), nn.Linear(hidden_dim//2, 1))

    def forward(self, z0, A, Z):
        B, H, N, d = Z.shape
        if H < 2: return torch.zeros(B, device=Z.device)
        Zp = self.proj(Z)
        early = Zp[:, :H//2].mean(dim=1); late = Zp[:, H//2:].mean(dim=1)
        per_slot = self.comp(torch.cat([early, late], dim=-1)).squeeze(-1)
        return self.aggregate(per_slot)


class SlotIWCMEnergy(nn.Module):
    def __init__(self, d_slot, d_action=11, hidden_dim=128, num_slots=8, lambdas=None):
        super().__init__()
        L = {"boundary": 1.0, "local": 1.0, "invariant": 1.5, "effect": 1.0, "counterfactual": 0.5}
        self.lambdas = {**L, **(lambdas or {})}
        self.boundary = SlotBoundaryHead(d_slot, d_action, hidden_dim, num_slots)
        self.local = SlotLocalHead(d_slot, d_action, hidden_dim, num_slots)
        self.invariant = SlotInvariantHead(d_slot, d_action, hidden_dim, num_slots)
        self.effect = SlotEffectHead(d_slot, d_action, hidden_dim, num_slots)
        self.counterfactual = SlotCounterfactualHead(d_slot, d_action, hidden_dim, num_slots)

    def forward(self, z0, A, Z):
        B = z0.shape[0]; e = torch.zeros(B, device=z0.device)
        for name, head, w in [("boundary", self.boundary, self.lambdas["boundary"]),
                               ("local", self.local, self.lambdas["local"]),
                               ("invariant", self.invariant, self.lambdas["invariant"]),
                               ("effect", self.effect, self.lambdas["effect"]),
                               ("counterfactual", self.counterfactual, self.lambdas["counterfactual"])]:
            e += w * head(z0, A, Z)
        return e

    def per_head(self, z0, A, Z):
        return {"boundary": self.boundary(z0, A, Z), "local": self.local(z0, A, Z),
                "invariant": self.invariant(z0, A, Z), "effect": self.effect(z0, A, Z),
                "counterfactual": self.counterfactual(z0, A, Z)}

    def score_acceptance(self, z0, A, Z):
        return torch.sigmoid(-self.forward(z0, A, Z))
