#!/usr/bin/env python3
"""Kill-shot: can a teacher validator transfer its detection signal to IWCM?

Trains mixed generalist teacher (SlotPool, ~0.939 AUROC), then tests 6 IWCM
training strategies:

  A: Oracle binary labels (upper bound, ~0.980)
  B: Raw pseudo binary labels (baseline, ~0.730)
  C: Teacher-weighted margin: weight high-invalid negatives more
  D: Teacher energy regression: E(Z) ≈ teacher_score
  E: Curriculum: crude→mixed→high-teacher-score
  F: Teacher-distilled ranking: margin scaled by teacher confidence

If any variant exceeds 0.85, the signal exists but needs the right loss.
If none do, self-supervised TAMG cannot train IWCM — hard stop.
"""
import sys, pickle, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from pathlib import Path
from collections import defaultdict
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.utils.seed import set_seed
from src.iwcm.fused_energy import FusedIWCMEnergy
from src.env.oracle_slot_encoder import ORACLE_SLOT_DIM, MAX_OBJECTS

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
D_SLOT = 19; N_SLOTS = 8; H_ = 25; D_ACTION = 11
BATCH = 32; EPOCHS = 150

with open("data/compositional_grid.pkl", "rb") as f:
    raw = pickle.load(f)

train_valid = [(torch.from_numpy(z0).float(), torch.from_numpy(A).float(),
                 torch.from_numpy(Z).float()) for z0, A, Z in raw["train_valid"]]
train_corr_raw = [(torch.from_numpy(e[0][0]).float(), torch.from_numpy(e[0][1]).float(),
                   torch.from_numpy(e[0][2]).float(), e[1]) for e in raw["train_corr"]]
test_valid  = [(torch.from_numpy(z0).float(), torch.from_numpy(A).float(),
                 torch.from_numpy(Z).float()) for z0, A, Z in raw["test_valid"]]
test_corr_raw = [(torch.from_numpy(e[0][0]).float(), torch.from_numpy(e[0][1]).float(),
                  torch.from_numpy(e[0][2]).float(), e[1]) for e in raw["test_corr"]]

test_by_type = defaultdict(list)
for z0, A, Z, meta in test_corr_raw:
    test_by_type[meta["violation_type"]].append((z0, A, Z))
VTYPES = sorted(test_by_type.keys())

train_corr_struct = [(z0, A, Z) for z0, A, Z, _ in train_corr_raw]

# ═══════════════════════════════════════════════════════════════════════════════
# Teacher: SlotPool mixed generalist
# ═══════════════════════════════════════════════════════════════════════════════

class SlotPoolValidator(nn.Module):
    def __init__(self):
        super().__init__()
        self.slot_encoder = nn.Sequential(nn.Linear(D_SLOT * 3, 128), nn.ReLU())
        self.cross_slot = nn.Sequential(
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.1), nn.Linear(128, 1))
    def forward(self, Z):
        B, H, Ns, d = Z.shape
        m = Z.mean(dim=1); mx = Z.amax(dim=1)
        v = F.relu((Z * Z).mean(dim=1) - m * m)
        Zp = torch.cat([m, mx, torch.sqrt(v + 1e-5)], dim=-1)
        sf = self.slot_encoder(Zp)
        return self.cross_slot(torch.cat([sf.mean(dim=1), sf.amax(dim=1)], dim=-1)).squeeze(-1)

# Generate mixed pseudo-negatives
valid_Z_stats = torch.stack([t[2] for t in train_valid])
valid_mean_norm = valid_Z_stats.norm(dim=-1).mean().item()
valid_std_norm = valid_Z_stats.norm(dim=-1).std().item()
valid_mean_active = (valid_Z_stats.norm(dim=-1) > 0.1).float().sum(dim=-1).float().mean().item()
valid_std_active = (valid_Z_stats.norm(dim=-1) > 0.1).float().sum(dim=-1).float().std().item()

def passes_filter(Zo, Zc):
    p = (Zc - Zo).pow(2).mean().sqrt().item()
    if p > 1.0: return False
    cn = Zc.norm(dim=-1)
    ac = (cn > 0.1).float().sum(dim=-1).float().mean().item()
    if ac < max(1.0, valid_mean_active - 3*valid_std_active): return False
    if ac > valid_mean_active + 3*valid_std_active: return False
    return True

mixed_corr = []
for z0, A, Z in train_valid:
    Zc = Z.clone()
    fam = np.random.randint(0, 8)
    if fam == 0:
        t = np.random.randint(1, H_); Zc[t:] = Zc[:H_ - t].clone()
    elif fam == 1 and N_SLOTS >= 2:
        i, j = np.random.choice(N_SLOTS, 2, replace=False)
        Zc[:, i], Zc[:, j] = Zc[:, j].clone(), Zc[:, i].clone()
    elif fam == 2:
        Zc[:, np.random.randint(0, N_SLOTS)] *= 0.0
    elif fam == 3 and N_SLOTS >= 2:
        i, j = np.random.choice(N_SLOTS, 2, replace=False)
        Zc[:, j] = Zc[:, i].clone()
    elif fam == 4:
        Zc[..., :5] += torch.randn_like(Zc[..., :5]) * 0.03
        Zc[..., 7:9] += torch.randn_like(Zc[..., 7:9]) * 0.03
    elif fam == 5:
        slot = np.random.randint(0, N_SLOTS)
        t = np.random.randint(1, H_ - 1)
        Zc[t:, slot, 5] += (np.random.rand() * 2 - 1) * 0.5
        Zc[t:, slot, 6] += (np.random.rand() * 2 - 1) * 0.5
    elif fam == 6:
        Zc += torch.randn_like(Zc) * 0.05
    elif fam == 7 and H_ >= 2:
        t = np.random.randint(0, H_ - 1)
        Zc[t+1:, np.random.randint(0, N_SLOTS)] += A[t].abs().mean().item() * 0.2
    if passes_filter(Z, Zc):
        mixed_corr.append((z0.clone(), A.clone(), Zc))

print(f"Train teacher on: {len(train_valid)} valid, {len(mixed_corr)} pseudo")

set_seed(42)
teacher = SlotPoolValidator().to(DEVICE)
opt = torch.optim.Adam(teacher.parameters(), lr=1e-3, weight_decay=1e-5)
for ep in range(100):
    vi = np.random.choice(len(train_valid), min(64, len(train_valid)), replace=False)
    ci = np.random.choice(len(mixed_corr), min(64, len(mixed_corr)), replace=False)
    Zv = torch.stack([train_valid[i][2] for i in vi]).to(DEVICE)
    Zc = torch.stack([mixed_corr[i][2] for i in ci]).to(DEVICE)
    Zb = torch.cat([Zv, Zc], dim=0)
    y = torch.cat([torch.zeros(len(vi)), torch.ones(len(ci))]).to(DEVICE)
    opt.zero_grad()
    loss = F.binary_cross_entropy_with_logits(teacher(Zb), y)
    loss.backward(); opt.step()
teacher.eval()

# Score teacher on train pseudo-negatives
with torch.no_grad():
    teacher_train_scores = []
    for _, _, Z in mixed_corr:
        teacher_train_scores.append(
            torch.sigmoid(teacher(Z.unsqueeze(0).to(DEVICE))).item())
teacher_train_scores = np.array(teacher_train_scores)
teacher_median = np.median(teacher_train_scores)
print(f"Teacher median score: {teacher_median:.4f}")

# Verify teacher AUROC
with torch.no_grad():
    tv = []
    for _, _, Z in test_valid:
        tv.append(torch.sigmoid(teacher(Z.unsqueeze(0).to(DEVICE))).item())
    tc = []
    for _, _, Z, _ in test_corr_raw:
        tc.append(torch.sigmoid(teacher(Z.unsqueeze(0).to(DEVICE))).item())
    t_auroc = roc_auc_score([0]*len(tv)+[1]*len(tc), tv+tc)
print(f"Teacher AUROC (oracle test): {t_auroc:.4f}")

# ═══════════════════════════════════════════════════════════════════════════════
# IWCM training variants
# ═══════════════════════════════════════════════════════════════════════════════

def make_iwcm():
    return FusedIWCMEnergy(d_slot=D_SLOT, d_action=D_ACTION, hidden=128,
                           num_slots=N_SLOTS).to(DEVICE)

def evaluate_iwcm(model):
    with torch.no_grad():
        ve = []
        for z0, A, Z in test_valid:
            ve.append(model(z0.unsqueeze(0).to(DEVICE), A.unsqueeze(0).to(DEVICE),
                           Z.unsqueeze(0).to(DEVICE)).item())
        per_type = {}
        for vt in VTYPES:
            ce = []
            for z0, A, Z in test_by_type[vt]:
                ce.append(model(z0.unsqueeze(0).to(DEVICE), A.unsqueeze(0).to(DEVICE),
                               Z.unsqueeze(0).to(DEVICE)).item())
            labels = [0]*len(ve) + [1]*len(ce)
            per_type[vt] = roc_auc_score(labels, ve + ce)
        all_ce = []
        for vt in VTYPES:
            for z0, A, Z in test_by_type[vt]:
                all_ce.append(model(z0.unsqueeze(0).to(DEVICE), A.unsqueeze(0).to(DEVICE),
                                    Z.unsqueeze(0).to(DEVICE)).item())
        all_labels = [0]*len(ve) + [1]*len(all_ce)
        per_type["OVERALL"] = roc_auc_score(all_labels, ve + all_ce)
    return per_type, ve

print("\n" + "=" * 75)
print("IWCM TRAINING: 6 strategies × 3 seeds")
print("=" * 75)

results_all = {}

def run_strategy(name, train_fn):
    print(f"\n  {name}")
    strat_results = {}
    for seed in [42, 123, 456]:
        torch.cuda.empty_cache()
        set_seed(seed)
        model = train_fn()
        r, _ = evaluate_iwcm(model)
        strat_results[seed] = r
        print(f"    seed={seed}: OVERALL={r['OVERALL']:.4f}  " +
              " ".join(f"{vt[:3]}={r[vt]:.3f}" for vt in VTYPES[:3]))
    avg = {}
    for vt in VTYPES + ["OVERALL"]:
        vals = [strat_results[s][vt] for s in [42, 123, 456]]
        avg[vt] = (np.mean(vals), np.std(vals))
    results_all[name] = avg
    return avg

# A: Oracle binary labels
def train_oracle():
    model = make_iwcm()
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    nv, nc = len(train_valid), len(train_corr_struct)
    for ep in range(EPOCHS):
        vi = np.random.choice(nv, min(BATCH, nv), replace=False)
        ci = np.random.choice(nc, min(BATCH, nc), replace=False)
        vz0 = torch.stack([train_valid[i][0] for i in vi]).to(DEVICE)
        vA  = torch.stack([train_valid[i][1] for i in vi]).to(DEVICE)
        vZ  = torch.stack([train_valid[i][2] for i in vi]).to(DEVICE)
        cz0 = torch.stack([train_corr_struct[i][0] for i in ci]).to(DEVICE)
        cA  = torch.stack([train_corr_struct[i][1] for i in ci]).to(DEVICE)
        cZ  = torch.stack([train_corr_struct[i][2] for i in ci]).to(DEVICE)
        opt.zero_grad()
        ev = model(vz0, vA, vZ); ec = model(cz0, cA, cZ)
        loss = F.relu(ev + 1.0).mean() + F.relu(1.0 - ec).mean() + \
               0.001*(ev.pow(2).mean() + ec.pow(2).mean())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
    return model.eval()

# B: Raw pseudo binary labels
def train_raw():
    model = make_iwcm()
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    nv, nc = len(train_valid), len(mixed_corr)
    for ep in range(EPOCHS):
        vi = np.random.choice(nv, min(BATCH, nv), replace=False)
        ci = np.random.choice(nc, min(BATCH, nc), replace=False)
        vz0 = torch.stack([train_valid[i][0] for i in vi]).to(DEVICE)
        vA  = torch.stack([train_valid[i][1] for i in vi]).to(DEVICE)
        vZ  = torch.stack([train_valid[i][2] for i in vi]).to(DEVICE)
        cz0 = torch.stack([mixed_corr[i][0] for i in ci]).to(DEVICE)
        cA  = torch.stack([mixed_corr[i][1] for i in ci]).to(DEVICE)
        cZ  = torch.stack([mixed_corr[i][2] for i in ci]).to(DEVICE)
        opt.zero_grad()
        ev = model(vz0, vA, vZ); ec = model(cz0, cA, cZ)
        loss = F.relu(ev + 1.0).mean() + F.relu(1.0 - ec).mean() + \
               0.001*(ev.pow(2).mean() + ec.pow(2).mean())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
    return model.eval()

# C: Teacher-weighted margin
def train_weighted():
    model = make_iwcm()
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    nv, nc = len(train_valid), len(mixed_corr)
    for ep in range(EPOCHS):
        vi = np.random.choice(nv, min(BATCH, nv), replace=False)
        ci = np.random.choice(nc, min(BATCH, nc), replace=False)
        vz0 = torch.stack([train_valid[i][0] for i in vi]).to(DEVICE)
        vA  = torch.stack([train_valid[i][1] for i in vi]).to(DEVICE)
        vZ  = torch.stack([train_valid[i][2] for i in vi]).to(DEVICE)
        cz0 = torch.stack([mixed_corr[i][0] for i in ci]).to(DEVICE)
        cA  = torch.stack([mixed_corr[i][1] for i in ci]).to(DEVICE)
        cZ  = torch.stack([mixed_corr[i][2] for i in ci]).to(DEVICE)
        weights = torch.tensor([teacher_train_scores[i] for i in ci],
                                device=DEVICE).clamp(0.1, 2.0)
        opt.zero_grad()
        ev = model(vz0, vA, vZ); ec = model(cz0, cA, cZ)
        loss = F.relu(ev + 1.0).mean() + \
               (weights * F.relu(1.0 - ec)).mean() + \
               0.001*(ev.pow(2).mean() + ec.pow(2).mean())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
    return model.eval()

# D: Teacher energy regression
def train_regression():
    model = make_iwcm()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    nv, nc = len(train_valid), len(mixed_corr)
    for ep in range(EPOCHS):
        vi = np.random.choice(nv, min(BATCH, nv), replace=False)
        ci = np.random.choice(nc, min(BATCH, nc), replace=False)
        vz0 = torch.stack([train_valid[i][0] for i in vi]).to(DEVICE)
        vA  = torch.stack([train_valid[i][1] for i in vi]).to(DEVICE)
        vZ  = torch.stack([train_valid[i][2] for i in vi]).to(DEVICE)
        cz0 = torch.stack([mixed_corr[i][0] for i in ci]).to(DEVICE)
        cA  = torch.stack([mixed_corr[i][1] for i in ci]).to(DEVICE)
        cZ  = torch.stack([mixed_corr[i][2] for i in ci]).to(DEVICE)
        Zb = torch.cat([vZ, cZ], dim=0)
        z0b = torch.cat([vz0, cz0], dim=0)
        Ab = torch.cat([vA, cA], dim=0)
        targets = torch.cat([
            torch.zeros(len(vi)),
            torch.tensor([teacher_train_scores[i] for i in ci], dtype=torch.float32)
        ]).to(DEVICE)
        opt.zero_grad()
        energies = model(z0b, Ab, Zb)
        loss = F.mse_loss(energies, targets * 5.0 - 1.0) + \
               0.01 * energies.pow(2).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
    return model.eval()

# E: Curriculum (crude → mixed → high-teacher)
def train_curriculum():
    model = make_iwcm()
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    nv = len(train_valid)

    # Phase 1: crude (easy negatives, epochs 0-49)
    crude_corr = []
    for z0, A, Z in train_valid:
        Zc = Z.clone() + torch.randn_like(Z) * 0.3
        Zc[:, :, :5] = Z[:, :, :5]
        Zc[:, :, 15:] = Z[:, :, 15:]
        crude_corr.append((z0.clone(), A.clone(), Zc))

    for ep in range(50):
        vi = np.random.choice(nv, min(BATCH, nv), replace=False)
        ci = np.random.choice(len(crude_corr), min(BATCH, len(crude_corr)), replace=False)
        vz0 = torch.stack([train_valid[i][0] for i in vi]).to(DEVICE)
        vA  = torch.stack([train_valid[i][1] for i in vi]).to(DEVICE)
        vZ  = torch.stack([train_valid[i][2] for i in vi]).to(DEVICE)
        cz0 = torch.stack([crude_corr[i][0] for i in ci]).to(DEVICE)
        cA  = torch.stack([crude_corr[i][1] for i in ci]).to(DEVICE)
        cZ  = torch.stack([crude_corr[i][2] for i in ci]).to(DEVICE)
        opt.zero_grad()
        ev = model(vz0, vA, vZ); ec = model(cz0, cA, cZ)
        loss = F.relu(ev + 1.0).mean() + F.relu(1.0 - ec).mean() + \
               0.001*(ev.pow(2).mean() + ec.pow(2).mean())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

    # Phase 2: mixed filtered (epochs 50-99)
    for ep in range(50, 100):
        vi = np.random.choice(nv, min(BATCH, nv), replace=False)
        ci = np.random.choice(len(mixed_corr), min(BATCH, len(mixed_corr)), replace=False)
        vz0 = torch.stack([train_valid[i][0] for i in vi]).to(DEVICE)
        vA  = torch.stack([train_valid[i][1] for i in vi]).to(DEVICE)
        vZ  = torch.stack([train_valid[i][2] for i in vi]).to(DEVICE)
        cz0 = torch.stack([mixed_corr[i][0] for i in ci]).to(DEVICE)
        cA  = torch.stack([mixed_corr[i][1] for i in ci]).to(DEVICE)
        cZ  = torch.stack([mixed_corr[i][2] for i in ci]).to(DEVICE)
        opt.zero_grad()
        ev = model(vz0, vA, vZ); ec = model(cz0, cA, cZ)
        loss = F.relu(ev + 1.0).mean() + F.relu(1.0 - ec).mean() + \
               0.001*(ev.pow(2).mean() + ec.pow(2).mean())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

    # Phase 3: high-teacher-score negatives (epochs 100-149)
    high_score_idx = np.where(teacher_train_scores > teacher_median)[0]
    high_corr = [mixed_corr[i] for i in high_score_idx]
    for ep in range(100, 150):
        vi = np.random.choice(nv, min(BATCH, nv), replace=False)
        ci = np.random.choice(len(high_corr), min(BATCH, len(high_corr)), replace=False)
        vz0 = torch.stack([train_valid[i][0] for i in vi]).to(DEVICE)
        vA  = torch.stack([train_valid[i][1] for i in vi]).to(DEVICE)
        vZ  = torch.stack([train_valid[i][2] for i in vi]).to(DEVICE)
        cz0 = torch.stack([high_corr[i][0] for i in ci]).to(DEVICE)
        cA  = torch.stack([high_corr[i][1] for i in ci]).to(DEVICE)
        cZ  = torch.stack([high_corr[i][2] for i in ci]).to(DEVICE)
        opt.zero_grad()
        ev = model(vz0, vA, vZ); ec = model(cz0, cA, cZ)
        loss = F.relu(ev + 1.0).mean() + F.relu(1.0 - ec).mean() + \
               0.001*(ev.pow(2).mean() + ec.pow(2).mean())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
    return model.eval()

# F: Teacher-distilled ranking
def train_ranking():
    model = make_iwcm()
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    nv, nc = len(train_valid), len(mixed_corr)
    for ep in range(EPOCHS):
        vi = np.random.choice(nv, min(BATCH, nv), replace=False)
        ci = np.random.choice(nc, min(BATCH, nc), replace=False)
        vz0 = torch.stack([train_valid[i][0] for i in vi]).to(DEVICE)
        vA  = torch.stack([train_valid[i][1] for i in vi]).to(DEVICE)
        vZ  = torch.stack([train_valid[i][2] for i in vi]).to(DEVICE)
        cz0 = torch.stack([mixed_corr[i][0] for i in ci]).to(DEVICE)
        cA  = torch.stack([mixed_corr[i][1] for i in ci]).to(DEVICE)
        cZ  = torch.stack([mixed_corr[i][2] for i in ci]).to(DEVICE)
        tw = torch.tensor([teacher_train_scores[i] for i in ci],
                           device=DEVICE)
        opt.zero_grad()
        ev = model(vz0, vA, vZ); ec = model(cz0, cA, cZ)
        margin = ec - ev
        loss = F.relu(ev + 1.0).mean() + \
               (tw * F.relu(2.0 - margin)).mean() + \
               0.001*(ev.pow(2).mean() + ec.pow(2).mean())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
    return model.eval()

# ═══════════════════════════════════════════════════════════════════════════════
# Run all strategies
# ═══════════════════════════════════════════════════════════════════════════════

run_strategy("A: Oracle binary          ", train_oracle)
run_strategy("B: Raw pseudo binary      ", train_raw)
run_strategy("C: Teacher-weighted       ", train_weighted)
run_strategy("D: Teacher regression     ", train_regression)
run_strategy("E: Curriculum             ", train_curriculum)
run_strategy("F: Teacher ranking margin ", train_ranking)

# ═══════════════════════════════════════════════════════════════════════════════
# Verdict
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 75)
print("FINAL VERDICT")
print("=" * 75)

header = f"{'Strategy':<28} {'OVERALL':>10}"
print(f"\n{header}")
print("-" * 40)
for name in results_all:
    mean, std = results_all[name]["OVERALL"]
    print(f"  {name:<26} {mean:8.4f} ± {std:.3f}")

oracle_mean = results_all["A: Oracle binary          "]["OVERALL"][0]
best_ss = max(results_all[n]["OVERALL"][0] for n in results_all
              if "Oracle" not in n)

print(f"\n  Oracle IWCM:                {oracle_mean:.4f}")
print(f"  Best self-supervised:       {best_ss:.4f}")
print(f"  Gap:                        {oracle_mean - best_ss:.4f}")
print(f"  Teacher AUROC:              {t_auroc:.4f}")

if best_ss >= 0.85:
    print(f"\n  KILL-SHOT PASSED: Teacher signal transfers to IWCM.")
    print(f"  Problem was training signal format, not signal quality.")
    print(f"  PATH: optimize teacher distillation loss for IWCM.")
elif best_ss >= 0.78:
    print(f"\n  MARGINAL: Modest improvement over raw pseudo (0.730).")
    print(f"  Teacher signal partially transfers but energy margin remains weak.")
    print(f"  PATH: stronger teacher or adaptive margin curriculum.")
else:
    print(f"\n  HARD STOP: Teacher signal does not transfer to IWCM.")
    print(f"  The IWCM energy landscape cannot be shaped by self-supervised")
    print(f"  validator signal — even when the validator is highly accurate.")
    print(f"  TAMG-as-IWCM-training is not viable with current methods.")
