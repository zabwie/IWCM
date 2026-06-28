"""SimpleTAMG — self-supervised corruption detection without symbolic oracle.

Wraps FusedIWCMEnergy (52K params). Trained via compositional mechanical
corruptions applied to oracle-slot tensors + anchored contrastive loss.

The oracle-slot format (19 channels) is a known, fixed encoding. Knowing
the channel layout is legitimate for Experiment 1 — the paper's contribution
is replacing the oracle's validity check with self-supervised training,
not building structure-agnostic corruptions.

Channel layout:
  0-4:  type one-hot    5-6:  position    7-8:  velocity
  9:    held flag       10:   visible     11-12: door/key
  13:   goal distance   14:   persistence 15:    existence
  16+:  id_hash
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.iwcm.fused_energy import FusedIWCMEnergy

POS = slice(5, 7); VEL = slice(7, 9); POS_VEL = slice(5, 9)
TYPE = slice(0, 5); EXIST = 15; ID_HASH = slice(16, None)


def _recompute_velocity(Z):
    Z[..., 1:, :, VEL] = Z[..., 1:, :, POS] - Z[..., :-1, :, POS]


def _corrupt(Z, rng):
    B, H, N, d = Z.shape
    Zc = Z.clone()
    for _ in range(rng.randint(2, 5)):
        op = rng.randint(0, 7)
        for i in range(B):
            ts = rng.randint(1, max(2, H - 2))
            s1 = rng.randint(0, N); s2 = rng.randint(0, N)
            while s2 == s1: s2 = rng.randint(0, N)
            e1 = Zc[i, ts, s1, EXIST].item() > 0.1
            e2 = Zc[i, ts, s2, EXIST].item() > 0.1

            if op == 0:
                tmp = Zc[i, ts:, s1, POS_VEL].clone()
                Zc[i, ts:, s1, POS_VEL] = Zc[i, ts:, s2, POS_VEL].clone()
                Zc[i, ts:, s2, POS_VEL] = tmp
            elif op == 1 and e1:
                pp = Zc[i, ts - 1, s1, POS].clone()
                Zc[i, ts:, s1, POS] = 0.0
                Zc[i, ts, s1, VEL] = -pp
                Zc[i, ts + 1:, s1, VEL] = 0.0
                Zc[i, ts:, s1, EXIST] = 0.0
            elif op == 2 and e1:
                dx = (rng.random() - 0.5) * 0.5
                dy = (rng.random() - 0.5) * 0.5
                Zc[i, ts:, s1, 5] += dx; Zc[i, ts:, s1, 6] += dy
                Zc[i, ts:, s1, 5].clamp_(0, 1); Zc[i, ts:, s1, 6].clamp_(0, 1)
                _recompute_velocity(Zc[i:i+1])
            elif op == 3 and e1 and not e2:
                Zc[i, :, s2] = Zc[i, :, s1].clone()
                Zc[i, :, s2, ID_HASH] = torch.rand(d - 16, device=Zc.device, dtype=Zc.dtype)
                Zc[i, :, s2, EXIST] = 1.0
            elif op == 4:
                sp = rng.randint(max(2, H // 3), max(3, 2 * H // 3))
                Zc[i, sp:] = Zc[i, sp:].flip(0)
                _recompute_velocity(Zc[i:i+1])
            elif op == 5 and e1:
                nt = torch.zeros(5, device=Zc.device, dtype=Zc.dtype)
                nt[rng.randint(0, 5)] = 1.0
                Zc[i, ts:, s1, TYPE] = nt
            elif op == 6:
                perm = list(range(N)); rng.shuffle(perm)
                Zc[i] = Zc[i, :, perm]
    return Zc


class SimpleTAMG(nn.Module):
    def __init__(self, d_slot=19, d_action=11, hidden=128, num_slots=8):
        super().__init__()
        self.energy_fn = FusedIWCMEnergy(
            d_slot=d_slot, d_action=d_action, hidden=hidden, num_slots=num_slots)
        self.rng = np.random.RandomState()

    def forward(self, z0, A, Z):
        return self.energy_fn(z0, A, Z)

    def score_acceptance(self, z0, A, Z):
        return torch.sigmoid(-self.forward(z0, A, Z))

    def training_step(self, z0, A, Z_valid, margin=1.0, reg=0.001):
        Z_corr = _corrupt(Z_valid, self.rng)
        diff = (Z_corr != Z_valid).any(dim=(-1, -2, -3))
        if diff.sum() < 2:
            return torch.tensor(0.0, device=Z_valid.device, requires_grad=True)
        z0_f, A_f = z0[diff], A[diff]
        Zv_f, Zc_f = Z_valid[diff], Z_corr[diff]
        Ev = self.energy_fn(z0_f, A_f, Zv_f)
        Ec = self.energy_fn(z0_f, A_f, Zc_f)
        loss = F.relu(Ev + 0.5).mean()
        loss = loss + F.relu(Ev + margin - Ec).mean()
        loss = loss + reg * (Ev.pow(2).mean() + Ec.pow(2).mean())
        return loss

    def evaluate(self, z0_list, A_list, Z_list):
        self.eval()
        scores = []
        with torch.no_grad():
            for z0, A, Z in zip(z0_list, A_list, Z_list):
                s = self.energy_fn(z0.unsqueeze(0), A.unsqueeze(0),
                                   Z.unsqueeze(0)).item()
                scores.append(s)
        return np.array(scores)


def load_compositional_grid(path="data/compositional_grid.pkl"):
    import pickle
    with open(path, "rb") as f:
        data = pickle.load(f)

    def _to_torch(items):
        return [(torch.from_numpy(z0).float(), torch.from_numpy(A).float(),
                 torch.from_numpy(Z).float()) for z0, A, Z in items]

    return {
        "train_valid": _to_torch(data["train_valid"]),
        "train_corr": [(torch.from_numpy(item[0][0]).float(),
                        torch.from_numpy(item[0][1]).float(),
                        torch.from_numpy(item[0][2]).float(), item[1])
                       for item in data["train_corr"]],
        "test_valid": _to_torch(data["test_valid"]),
        "test_corr": [(torch.from_numpy(item[0][0]).float(),
                       torch.from_numpy(item[0][1]).float(),
                       torch.from_numpy(item[0][2]).float(), item[1])
                      for item in data["test_corr"]],
    }
