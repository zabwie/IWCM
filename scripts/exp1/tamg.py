#!/usr/bin/env python3
"""TAMG v3: logit-based validators + adversarial operator tuning + self-supervised IWCM."""
import sys, pickle, numpy as np, torch, torch.nn.functional as F, types
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.seed import set_seed
from src.iwcm.slot_energy import SlotIWCMEnergy
from src.tamg.validators.committee import Validator
from src.encoder.oracle_slot_encoder import ORACLE_SLOT_DIM, MAX_OBJECTS
from collections import defaultdict
from sklearn.metrics import roc_auc_score
from sklearn.cluster import KMeans

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
set_seed(42)

with open("data/compositional_grid.pkl", "rb") as f:
    data = pickle.load(f)
tv = [(torch.from_numpy(z0).float(), torch.from_numpy(A).float(), torch.from_numpy(Z).float())
      for z0, A, Z in data["train_valid"]]
test_valid = [(torch.from_numpy(z0).float(), torch.from_numpy(A).float(), torch.from_numpy(Z).float())
              for z0, A, Z in data["test_valid"]]
test_corr = [(torch.from_numpy(item[0][0]).float(), torch.from_numpy(item[0][1]).float(),
              torch.from_numpy(item[0][2]).float(), item[1]) for item in data["test_corr"]]
test_by_type = defaultdict(list)
for z0, A, Z, meta in test_corr:
    test_by_type[meta["violation_type"]].append((z0, A, Z))
N, d, H = MAX_OBJECTS, ORACLE_SLOT_DIM, 25

# ─── STEP 1: Cluster diffs ────────────────────────────────────────────────
diffs_list = []
for z0, A, Z in tv[:50]:
    for t in range(H - 1):
        for n in range(N):
            if Z[t, n].sum() > 0.1:
                diffs_list.append((Z[t + 1, n] - Z[t, n]).numpy())
diffs_arr = np.stack(diffs_list)
n_ops = 8
kmeans = KMeans(n_clusters=n_ops, random_state=42, n_init=10).fit(diffs_arr)
centroids = torch.from_numpy(kmeans.cluster_centers_).float().to(DEVICE)
print(f"STEP1: {len(diffs_arr)} diffs -> {n_ops} centroids")

# ─── STEP 2: 6 logit-based validators ──────────────────────────────────────

class V1(Validator):
    name = "v1_local"

class V2(Validator):
    name = "v2_velocity"

class V3(Validator):
    name = "v3_global"

class V4(Validator):
    name = "v4_cross"

class V5(Validator):
    name = "v5_short"

class V6(Validator):
    name = "v6_gating"

validators = [V1(), V2(), V3(), V4(), V5(), V6()]

# V1: local transition MLP
validators[0].mlp = torch.nn.Sequential(
    torch.nn.Linear(d * 2, 128), torch.nn.ReLU(), torch.nn.Linear(128, 1))

def v1_fwd(self, Z, z0=None, A=None):
    return self.mlp(torch.cat(
        [Z[:, :-1].reshape(-1, N, d), Z[:, 1:].reshape(-1, N, d)], dim=-1
    )).squeeze(-1).reshape(Z.shape[0], -1, N).mean(dim=(-2, -1))

validators[0].forward = types.MethodType(v1_fwd, validators[0])

# V2: velocity-based (structurally different from V1 — operates on diffs)
validators[1].mlp = torch.nn.Sequential(
    torch.nn.Linear(d, 64), torch.nn.ReLU(), torch.nn.Linear(64, 1))

def v2_fwd(self, Z, z0=None, A=None):
    return self.mlp(
        (Z[:, 1:] - Z[:, :-1]).reshape(-1, N, d)
    ).squeeze(-1).reshape(Z.shape[0], -1, N).mean(dim=(-2, -1))

validators[1].forward = types.MethodType(v2_fwd, validators[1])

# V3: bidirectional GRU (global structure, different from MLP-based validators)
validators[2].gru = torch.nn.GRU(d, 64, bidirectional=True, batch_first=True)
validators[2].scorer = torch.nn.Linear(128, 1)

def v3_fwd(self, Z, z0=None, A=None):
    B = Z.shape[0]
    _, h = self.gru(Z.reshape(B, -1, d))
    return self.scorer(torch.cat([h[0], h[1]], dim=-1)).squeeze(-1)

validators[2].forward = types.MethodType(v3_fwd, validators[2])

# V4: cross-time contrast (Δt=H/2, different horizon)
validators[3].proj = torch.nn.Linear(d, 32)
validators[3].scorer = torch.nn.Linear(32, 1)

def v4_fwd(self, Z, z0=None, A=None):
    B, H, N_in, d_in = Z.shape
    half = H // 2
    if half < 1:
        return torch.zeros(B, device=Z.device)
    return self.scorer(
        (self.proj(Z[:, :half].mean(dim=1)) - self.proj(Z[:, half:].mean(dim=1))).abs()
    ).squeeze(-1).mean(dim=-1)

validators[3].forward = types.MethodType(v4_fwd, validators[3])

# V5: short-range (first 5 frames only, different horizon)
validators[4].mlp = torch.nn.Sequential(
    torch.nn.Linear(d * 2, 64), torch.nn.ReLU(), torch.nn.Linear(64, 1))

def v5_fwd(self, Z, z0=None, A=None):
    B, H, N_in, d_in = Z.shape
    h = min(5, H)
    if h < 2:
        return torch.zeros(B, device=Z.device)
    return self.mlp(torch.cat(
        [Z[:, :h - 1].reshape(-1, N, d), Z[:, 1:h].reshape(-1, N, d)], dim=-1
    )).squeeze(-1).reshape(B, h - 1, N).mean(dim=(-2, -1))

validators[4].forward = types.MethodType(v5_fwd, validators[4])

# V6: gating (slot activity detector — parameter-free style)
validators[5].gate = torch.nn.Sequential(torch.nn.Linear(d, 1), torch.nn.Sigmoid())

def v6_fwd(self, Z, z0=None, A=None):
    return self.gate(Z).squeeze(-1).mean(dim=(1, 2))

validators[5].forward = types.MethodType(v6_fwd, validators[5])

for v in validators:
    v.to(DEVICE)

params = [p for v in validators for p in v.parameters()]
opt_v = torch.optim.Adam(params, lr=1e-3)
for ep in range(150):
    vi = np.random.choice(len(tv), 32, replace=False)
    Z_b = torch.stack([tv[i][2] for i in vi]).to(DEVICE)
    z0_b = torch.stack([tv[i][0] for i in vi]).to(DEVICE)
    A_b = torch.stack([tv[i][1] for i in vi]).to(DEVICE)
    loss = 0
    for v in validators:
        logits = v(Z_b, z0_b, A_b)
        loss += F.binary_cross_entropy_with_logits(logits, torch.ones_like(logits))
    opt_v.zero_grad()
    loss.backward()
    opt_v.step()
    if (ep + 1) % 75 == 0:
        print(f"  ep{ep+1}: loss={loss.item():.4f}")

for v in validators:
    v.eval()
print(f"STEP2: {len(validators)} validators trained")

# ─── STEP 3: Adversarial operator tuning (maximize validator spread) ───────
print("\nSTEP3: Adversarial operator tuning...")
for v in validators:
    v.train()  # GRU needs training mode for backward

operators = centroids.clone().requires_grad_(True)
opt_op = torch.optim.Adam([operators], lr=0.02)

for ep in range(300):
    vi = np.random.choice(len(tv), 16, replace=False)
    Z_b = torch.stack([tv[i][2] for i in vi]).to(DEVICE)
    z0_b = torch.stack([tv[i][0] for i in vi]).to(DEVICE)
    A_b = torch.stack([tv[i][1] for i in vi]).to(DEVICE)
    B = Z_b.shape[0]

    total_spread = 0
    total_valid_agree = 0
    for k in range(n_ops):
        Z_c = Z_b.clone()
        for b in range(B):
            Z_c[b, np.random.randint(0, max(1, H // 2)):, np.random.randint(0, N)] += operators[k].unsqueeze(0) * 0.3

        v_logits = [v(Z_c, z0_b, A_b) for v in validators]
        total_spread += torch.stack(v_logits, dim=0).var(dim=0).mean()

    # Also measure spread on VALID trajectories (want LOW spread = agreement)
    v_logits_valid = [v(Z_b, z0_b, A_b) for v in validators]
    total_valid_agree += torch.stack(v_logits_valid, dim=0).var(dim=0).mean()

    spread = total_spread / n_ops
    valid_spread = total_valid_agree
    # Maximize corrupt spread, minimize valid spread, stay near centroids
    loss = -spread + 2.0 * valid_spread + 0.05 * ((operators - centroids) ** 2).sum(dim=-1).mean()
    opt_op.zero_grad()
    loss.backward()
    opt_op.step()
    with torch.no_grad():
        operators.data = centroids + (operators - centroids).clamp(-0.5, 0.5)

    if (ep + 1) % 150 == 0:
        print(f"  ep{ep+1}: spread_c={spread.item():.4f} spread_v={valid_spread.item():.4f}")

operators = operators.detach()
for v in validators:
    v.eval()
print(f"  Final spread: {spread.item():.4f}")

# ─── STEP 4: Self-supervised IWCM training ────────────────────────────────
print("\nSTEP4: Self-supervised IWCM...")
model = SlotIWCMEnergy(d_slot=d, d_action=11, hidden_dim=128, num_slots=N).to(DEVICE)
opt = torch.optim.Adam(model.parameters(), lr=3e-3)

def disagreement_sigmoid(Z_b, z0_b, A_b):
    probs = []
    for v in validators:
        with torch.no_grad():
            logits = v(Z_b, z0_b, A_b)
            probs.append(torch.sigmoid(logits))
    stacked = torch.stack(probs, dim=0)
    return stacked.var(dim=0)

for ep in range(300):
    batch = min(32, len(tv))
    vi = np.random.choice(len(tv), batch, replace=False)
    vZ = torch.stack([tv[i][2] for i in vi]).to(DEVICE)
    vz0 = torch.stack([tv[i][0] for i in vi]).to(DEVICE)
    vA = torch.stack([tv[i][1] for i in vi]).to(DEVICE)

    cZ = vZ.clone()
    for b in range(batch):
        cZ[b, np.random.randint(0, max(1, H // 2)):, np.random.randint(0, N)] += operators[np.random.randint(0, n_ops)] * 0.4

    d_c = disagreement_sigmoid(cZ, vz0, vA)
    d_v = disagreement_sigmoid(vZ, vz0, vA)

    ev = model(vz0, vA, vZ)
    ec = model(vz0, vA, cZ)
    w = (d_c - d_v).detach().clamp(0.0, 1.0) + 0.1
    loss = F.relu(ev + 1).mean() + (w * F.relu(1 - ec)).mean() + 0.001 * (ev.pow(2).mean() + ec.pow(2).mean())
    opt.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    if (ep + 1) % 100 == 0:
        print(f"  ep{ep+1}: ev={ev.mean():+.2f} ec={ec.mean():+.2f} d_c={d_c.mean():.4f} d_v={d_v.mean():.4f}")

# ─── STEP 5: Evaluate ─────────────────────────────────────────────────────
model.eval()
print(f"\nSTEP5: Cross-surface evaluation...")
with torch.no_grad():
    ve = [model(z0.unsqueeze(0).to(DEVICE), A.unsqueeze(0).to(DEVICE),
                Z.unsqueeze(0).to(DEVICE)).item() for z0, A, Z in test_valid]
    per_type = {}
    for vt in sorted(test_by_type.keys()):
        ce = [model(z0.unsqueeze(0).to(DEVICE), A.unsqueeze(0).to(DEVICE),
                    Z.unsqueeze(0).to(DEVICE)).item() for z0, A, Z in test_by_type[vt]]
        a = roc_auc_score([0] * len(ve) + [1] * len(ce), ve + ce)
        per_type[vt] = a
        print(f"  {vt:<14} {a:.3f}")

avg = np.mean(list(per_type.values()))
print(f"  {'AVERAGE':<14} {avg:.3f}")
print(f"\nOracle: 0.962 | Random: 0.734 | TAMG v1: 0.748 | TAMG v3: {avg:.3f}")
