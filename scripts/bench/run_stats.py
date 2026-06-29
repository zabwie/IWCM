#!/usr/bin/env python3
"""5-seed statistical significance + per-axis generalization matrix."""
import sys, torch, pickle, numpy as np, torch.nn.functional as F
from collections import defaultdict
sys.path.insert(0, '.')

from src.utils.seed import set_seed
from src.env.oracle_slot_encoder import ORACLE_SLOT_DIM, MAX_OBJECTS

H, N, d = 25, MAX_OBJECTS, ORACLE_SLOT_DIM
device = "cuda" if torch.cuda.is_available() else "cpu"

with open("data/compositional_grid.pkl", "rb") as f:
    grid = pickle.load(f)


def slot_data(entries):
    return [(torch.from_numpy(z0).float(), torch.from_numpy(A).float(),
             torch.from_numpy(Z).float()) for z0, A, Z in entries]


class FlatMLP(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(H * N * d, 256), torch.nn.ReLU(),
            torch.nn.Linear(256, 64), torch.nn.ReLU(),
            torch.nn.Linear(64, 1))

    def forward(self, z0, A, Z):
        return self.net(Z.reshape(Z.shape[0], -1)).squeeze(-1)


class SlotTransformer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = torch.nn.Linear(d, 64)
        enc = torch.nn.TransformerEncoderLayer(
            d_model=64, nhead=4, dim_feedforward=128,
            dropout=0.1, batch_first=True)
        self.transformer = torch.nn.TransformerEncoder(enc, num_layers=2)
        self.scorer = torch.nn.Linear(64, 1)

    def forward(self, z0, A, Z):
        B = Z.shape[0]
        Zp = self.proj(Z).reshape(B, H * N, 64)
        return self.scorer(self.transformer(Zp).mean(dim=1)).squeeze(-1)


def train_eval(ModelClass, train_v, train_c_meta, test_v, test_c_meta,
               epochs=150, lr=3e-4, seed=42):
    set_seed(seed)
    model = ModelClass().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for ep in range(epochs):
        vi = np.random.choice(len(train_v), min(16, len(train_v)), replace=False)
        ci = np.random.choice(len(train_c_meta), min(32, len(train_c_meta)), replace=False)
        vz0 = torch.stack([train_v[i][0] for i in vi]).to(device)
        vA = torch.stack([train_v[i][1] for i in vi]).to(device)
        vZ = torch.stack([train_v[i][2] for i in vi]).to(device)
        cz0 = torch.stack([train_c_meta[i][0] for i in ci]).to(device)
        cA = torch.stack([train_c_meta[i][1] for i in ci]).to(device)
        cZ = torch.stack([train_c_meta[i][2] for i in ci]).to(device)
        opt.zero_grad()
        ev = model(vz0, vA, vZ)
        ec = model(cz0, cA, cZ)
        loss = (F.relu(ev + 1.0).mean() + F.relu(1.0 - ec).mean() +
                0.001 * (ev.pow(2).mean() + ec.pow(2).mean()))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

    per_law = defaultdict(list)
    v_acc = []
    i_rej = []
    for vz, vA, vZ in test_v[:80]:
        with torch.no_grad():
            s = torch.sigmoid(-model(
                vz.unsqueeze(0).to(device), vA.unsqueeze(0).to(device),
                vZ.unsqueeze(0).to(device))).item()
        v_acc.append(s > 0.5)
    for cz, cA, cZ, meta in test_c_meta:
        with torch.no_grad():
            s = torch.sigmoid(-model(
                cz.unsqueeze(0).to(device), cA.unsqueeze(0).to(device),
                cZ.unsqueeze(0).to(device))).item()
        i_rej.append(s < 0.5)
        per_law[meta["law_type"]].append(s < 0.5)

    return {
        "valid_acc": np.mean(v_acc),
        "invalid_rej": np.mean(i_rej),
        "conservation": np.mean(per_law.get("conservation", [0.])),
        "identity": np.mean(per_law.get("identity", [0.])),
    }


# Prep data
tv = slot_data(grid["train_valid"])
tcm = [(torch.from_numpy(e[0]).float(), torch.from_numpy(e[1]).float(),
        torch.from_numpy(e[2]).float(), m) for e, m in grid["train_corr"]]
tev = slot_data(grid["test_valid"])
tecm = [(torch.from_numpy(e[0]).float(), torch.from_numpy(e[1]).float(),
         torch.from_numpy(e[2]).float(), m) for e, m in grid["test_corr"]]

# 5-seed comparison
models = {"MLP": FlatMLP, "SlotTransformer": SlotTransformer}
seeds = [42, 123, 456, 789, 1011]
print("=" * 70)
print("5-SEED STATISTICAL SIGNIFICANCE")
print("=" * 70)

for name, cls in models.items():
    results = []
    for s in seeds:
        r = train_eval(cls, tv, tcm, tev, tecm, seed=s, epochs=100)
        results.append(r)
    keys = ["valid_acc", "invalid_rej", "conservation", "identity"]
    means = {k: np.mean([r[k] for r in results]) for k in keys}
    stds = {k: np.std([r[k] for r in results]) for k in keys}
    print(f"{name}:")
    print(f"  valid_acc    = {means['valid_acc']:.3f} ± {stds['valid_acc']:.3f}")
    print(f"  invalid_rej  = {means['invalid_rej']:.3f} ± {stds['invalid_rej']:.3f}")
    print(f"  conservation = {means['conservation']:.3f} ± {stds['conservation']:.3f}")
    print(f"  identity     = {means['identity']:.3f} ± {stds['identity']:.3f}")

# Per-axis generalization
print("\n" + "=" * 70)
print("PER-AXIS GENERALIZATION MATRIX (MLP)")
print("=" * 70)

set_seed(42)
mlp = FlatMLP().to(device)
opt = torch.optim.Adam(mlp.parameters(), lr=3e-4)
for ep in range(150):
    vi = np.random.choice(len(tv), min(16, len(tv)), replace=False)
    ci = np.random.choice(len(tcm), min(32, len(tcm)), replace=False)
    vz0 = torch.stack([tv[i][0] for i in vi]).to(device)
    vA = torch.stack([tv[i][1] for i in vi]).to(device)
    vZ = torch.stack([tv[i][2] for i in vi]).to(device)
    cz0 = torch.stack([tcm[i][0] for i in ci]).to(device)
    cA = torch.stack([tcm[i][1] for i in ci]).to(device)
    cZ = torch.stack([tcm[i][2] for i in ci]).to(device)
    opt.zero_grad()
    ev = mlp(vz0, vA, vZ); ec = mlp(cz0, cA, cZ)
    loss = (F.relu(ev + 1.0).mean() + F.relu(1.0 - ec).mean() +
            0.001 * (ev.pow(2).mean() + ec.pow(2).mean()))
    loss.backward()
    torch.nn.utils.clip_grad_norm_(mlp.parameters(), 1.0)
    opt.step()

by_object = defaultdict(list)
by_context = defaultdict(list)
by_violation = defaultdict(list)
by_gap = defaultdict(list)
for cz, cA, cZ, meta in tecm:
    with torch.no_grad():
        s = torch.sigmoid(-mlp(
            cz.unsqueeze(0).to(device), cA.unsqueeze(0).to(device),
            cZ.unsqueeze(0).to(device))).item()
    by_object[meta["object_type"]].append(s < 0.5)
    by_context[meta["context"]].append(s < 0.5)
    by_violation[meta["violation_type"]].append(s < 0.5)
    by_gap[meta["time_gap"]].append(s < 0.5)

for axis_name, axis_data in [
    ("Object Type", by_object), ("Context", by_context),
    ("Violation Type", by_violation), ("Time Gap", by_gap),
]:
    print(f"  {axis_name}:", end="")
    for k in sorted(axis_data.keys()):
        print(f" {k}={np.mean(axis_data[k]):.3f}", end="")
    print()
