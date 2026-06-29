#!/usr/bin/env python3
"""TAMG-v2: Teacher-guided adversarial counterexample mining

The only surviving mechanism: a validator identifies invalidity, and an
adversarial search finds IWCM-boundary failures that the validator approves.

Pipeline:
  1. Train teacher validator (mixed SlotPool, ~0.91 AUROC)
  2. Warm up IWCM on crude easy negatives (50 epochs)
  3. Adversarial mining loop (100 epochs):
     a. Generate mixed pseudo-negatives
     b. Gradient-ascent on (teacher_score - IWCM_energy) to find boundary examples
     c. Train IWCM with teacher-weighted margin on mined negatives

Compares: A(oracle), B(raw pseudo), C(teacher-weighted), D(teacher regression),
          E(curriculum), F(adversarial mining)
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
BATCH = 32; EPOCHS = 150; WARMUP = 50

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

valid_Zs = torch.stack([t[2] for t in train_valid])
valid_mean_active = (valid_Zs.norm(dim=-1) > 0.1).float().sum(dim=-1).float().mean().item()
valid_std_active = (valid_Zs.norm(dim=-1) > 0.1).float().sum(dim=-1).float().std().item()

print(f"Train: {len(train_valid)}v + {len(train_corr_struct)}c")
print(f"Test:  {len(test_valid)}v + {len(test_corr_raw)}c")

# ═══════════════════════════════════════════════════════════════════════════════
# Teacher (SlotPool mixed generalist)
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

def passes_filter(Zo, Zc):
    p = (Zc - Zo).pow(2).mean().sqrt().item()
    if p > 1.0: return False
    cn = Zc.norm(dim=-1)
    ac = (cn > 0.1).float().sum(dim=-1).float().mean().item()
    if ac < max(1.0, valid_mean_active - 3*valid_std_active): return False
    if ac > valid_mean_active + 3*valid_std_active: return False
    return True

def generate_mixed(valid_data):
    corr = []
    for z0, A, Z in valid_data:
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
            corr.append((z0.clone(), A.clone(), Zc))
    return corr

print("\nTraining teacher...")
mixed_corr = generate_mixed(train_valid)
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
for p in teacher.parameters(): p.requires_grad = False

# Verify teacher
with torch.no_grad():
    tv = [torch.sigmoid(teacher(Z.unsqueeze(0).to(DEVICE))).item()
          for _, _, Z in test_valid]
    tc_full = [torch.sigmoid(teacher(Z.unsqueeze(0).to(DEVICE))).item()
               for _, _, Z, _ in test_corr_raw]
t_auroc = roc_auc_score([0]*len(tv)+[1]*len(tc_full), tv+tc_full)
print(f"Teacher AUROC: {t_auroc:.4f}")

# ═══════════════════════════════════════════════════════════════════════════════
# IWCM + evaluation
# ═══════════════════════════════════════════════════════════════════════════════

def evaluate_iwcm(model):
    with torch.no_grad():
        ve = [model(z0.unsqueeze(0).to(DEVICE), A.unsqueeze(0).to(DEVICE),
                    Z.unsqueeze(0).to(DEVICE)).item() for z0, A, Z in test_valid]
        per_type = {}
        for vt in VTYPES:
            ce = [model(z0.unsqueeze(0).to(DEVICE), A.unsqueeze(0).to(DEVICE),
                        Z.unsqueeze(0).to(DEVICE)).item() for z0, A, Z in test_by_type[vt]]
            per_type[vt] = roc_auc_score([0]*len(ve)+[1]*len(ce), ve+ce)
        all_ce = []
        for vt in VTYPES:
            for z0, A, Z in test_by_type[vt]:
                all_ce.append(model(z0.unsqueeze(0).to(DEVICE), A.unsqueeze(0).to(DEVICE),
                                   Z.unsqueeze(0).to(DEVICE)).item())
        per_type["OVERALL"] = roc_auc_score([0]*len(ve)+[1]*len(all_ce), ve+all_ce)
    return per_type

# ═══════════════════════════════════════════════════════════════════════════════
# TAMG-v2: Adversarial counterexample mining
# ═══════════════════════════════════════════════════════════════════════════════

def train_tamg_v2(seed):
    set_seed(seed)
    model = FusedIWCMEnergy(d_slot=D_SLOT, d_action=D_ACTION, hidden=128,
                            num_slots=N_SLOTS).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)

    # Phase 1: warm up on crude easy negatives
    crude_corr = []
    for z0, A, Z in train_valid:
        Zc = Z.clone() + torch.randn_like(Z) * 0.3
        Zc[:, :, :5] = Z[:, :, :5]
        Zc[:, :, 15:] = Z[:, :, 15:]
        crude_corr.append((z0.clone(), A.clone(), Zc))

    nv = len(train_valid)
    for ep in range(WARMUP):
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

    # Phase 2: adversarial mining loop
    for ep in range(WARMUP, EPOCHS):
        # Generate fresh mixed pseudo-negatives pool
        pool_corr = generate_mixed(train_valid)
        if len(pool_corr) < BATCH:
            continue

        # Score pool with teacher and IWCM
        pool_Z = torch.stack([p[2] for p in pool_corr]).to(DEVICE)
        pool_z0 = torch.stack([p[0] for p in pool_corr]).to(DEVICE)
        pool_A  = torch.stack([p[1] for p in pool_corr]).to(DEVICE)

        with torch.no_grad():
            t_scores = torch.sigmoid(teacher(pool_Z)).cpu().numpy()
            iwcm_energies = model(pool_z0, pool_A, pool_Z).cpu().numpy()

        # Select negatives: teacher says invalid AND IWCM energy < threshold
        invalid_mask = t_scores > 0.4
        low_energy_mask = iwcm_energies < 1.0
        good_idx = np.where(invalid_mask & low_energy_mask)[0]

        if len(good_idx) < BATCH // 2:
            good_idx = np.where(invalid_mask)[0]
        if len(good_idx) < BATCH // 2:
            good_idx = np.arange(len(pool_corr))

        # Take batch and do gradient ascent to make them harder for IWCM
        chosen = np.random.choice(good_idx, min(BATCH, len(good_idx)), replace=False)
        cZ_batch = pool_Z[chosen].clone().detach().requires_grad_(True)
        cz0_batch = pool_z0[chosen]
        cA_batch = pool_A[chosen]

        # Gradient ascent: maximize teacher_invalid - IWCM_energy
        adv_opt = torch.optim.Adam([cZ_batch], lr=0.01)
        for _ in range(3):
            adv_opt.zero_grad()
            t_adv = torch.sigmoid(teacher(cZ_batch))
            e_adv = model(cz0_batch, cA_batch, cZ_batch)
            # Maximize: teacher_score - energy
            adv_loss = -(t_adv.mean() - 0.3 * e_adv.mean())
            adv_loss.backward()
            adv_opt.step()
            # Clamp to manifold
            cZ_batch.data = cZ_batch.data.clamp(-5.0, 5.0)

        cZ_final = cZ_batch.detach()

        # Train IWCM on valid + adversarial negatives with teacher weights
        vi = np.random.choice(nv, min(BATCH, nv), replace=False)
        vz0 = torch.stack([train_valid[i][0] for i in vi]).to(DEVICE)
        vA  = torch.stack([train_valid[i][1] for i in vi]).to(DEVICE)
        vZ  = torch.stack([train_valid[i][2] for i in vi]).to(DEVICE)

        with torch.no_grad():
            final_t = torch.sigmoid(teacher(cZ_final)).clamp(0.1, 2.0)

        opt.zero_grad()
        ev = model(vz0, vA, vZ)
        ec = model(cz0_batch, cA_batch, cZ_final)
        loss = F.relu(ev + 1.0).mean() + \
               (final_t * F.relu(1.0 - ec)).mean() + \
               0.001*(ev.pow(2).mean() + ec.pow(2).mean())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

    return model.eval()

# ═══════════════════════════════════════════════════════════════════════════════
# Run and compare
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 75)
print("TAMG-v2: 6 strategies × 3 seeds")
print("=" * 75)

results_all = {}

def run_strat(name, fn):
    print(f"\n  {name}")
    r = {}
    for seed in [42, 123, 456]:
        torch.cuda.empty_cache()
        set_seed(seed)
        model = fn(seed)
        res = evaluate_iwcm(model)
        r[seed] = res
        print(f"    seed={seed}: OVERALL={res['OVERALL']:.4f}  " +
              " ".join(f"{vt[:3]}={res[vt]:.3f}" for vt in VTYPES[:3]))
    avg = {}
    for vt in VTYPES + ["OVERALL"]:
        vals = [r[s][vt] for s in [42, 123, 456]]
        avg[vt] = (np.mean(vals), np.std(vals))
    results_all[name] = avg
    return avg

# A: Oracle
def train_oracle(seed):
    set_seed(seed)
    model = FusedIWCMEnergy(d_slot=D_SLOT, d_action=D_ACTION, hidden=128,
                            num_slots=N_SLOTS).to(DEVICE)
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

# B: Raw pseudo
def train_raw(seed):
    set_seed(seed)
    model = FusedIWCMEnergy(d_slot=D_SLOT, d_action=D_ACTION, hidden=128,
                            num_slots=N_SLOTS).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    nv = len(train_valid)
    for ep in range(EPOCHS):
        corr = generate_mixed(train_valid)
        nc = len(corr)
        vi = np.random.choice(nv, min(BATCH, nv), replace=False)
        ci = np.random.choice(nc, min(BATCH, nc), replace=False)
        vz0 = torch.stack([train_valid[i][0] for i in vi]).to(DEVICE)
        vA  = torch.stack([train_valid[i][1] for i in vi]).to(DEVICE)
        vZ  = torch.stack([train_valid[i][2] for i in vi]).to(DEVICE)
        cz0 = torch.stack([corr[i][0] for i in ci]).to(DEVICE)
        cA  = torch.stack([corr[i][1] for i in ci]).to(DEVICE)
        cZ  = torch.stack([corr[i][2] for i in ci]).to(DEVICE)
        opt.zero_grad()
        ev = model(vz0, vA, vZ); ec = model(cz0, cA, cZ)
        loss = F.relu(ev + 1.0).mean() + F.relu(1.0 - ec).mean() + \
               0.001*(ev.pow(2).mean() + ec.pow(2).mean())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
    return model.eval()

# C: Teacher-weighted
def train_weighted(seed):
    set_seed(seed)
    model = FusedIWCMEnergy(d_slot=D_SLOT, d_action=D_ACTION, hidden=128,
                            num_slots=N_SLOTS).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    nv = len(train_valid)
    for ep in range(EPOCHS):
        corr = generate_mixed(train_valid)
        nc = len(corr)
        vi = np.random.choice(nv, min(BATCH, nv), replace=False)
        ci = np.random.choice(nc, min(BATCH, nc), replace=False)
        vz0 = torch.stack([train_valid[i][0] for i in vi]).to(DEVICE)
        vA  = torch.stack([train_valid[i][1] for i in vi]).to(DEVICE)
        vZ  = torch.stack([train_valid[i][2] for i in vi]).to(DEVICE)
        cz0 = torch.stack([corr[i][0] for i in ci]).to(DEVICE)
        cA  = torch.stack([corr[i][1] for i in ci]).to(DEVICE)
        cZ  = torch.stack([corr[i][2] for i in ci]).to(DEVICE)
        with torch.no_grad():
            tw = torch.sigmoid(teacher(cZ)).clamp(0.1, 2.0)
        opt.zero_grad()
        ev = model(vz0, vA, vZ); ec = model(cz0, cA, cZ)
        loss = F.relu(ev + 1.0).mean() + (tw * F.relu(1.0 - ec)).mean() + \
               0.001*(ev.pow(2).mean() + ec.pow(2).mean())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
    return model.eval()

# D: Teacher regression
def train_regression(seed):
    set_seed(seed)
    model = FusedIWCMEnergy(d_slot=D_SLOT, d_action=D_ACTION, hidden=128,
                            num_slots=N_SLOTS).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    nv = len(train_valid)
    for ep in range(EPOCHS):
        corr = generate_mixed(train_valid)
        nc = len(corr)
        vi = np.random.choice(nv, min(BATCH, nv), replace=False)
        ci = np.random.choice(nc, min(BATCH, nc), replace=False)
        vz0 = torch.stack([train_valid[i][0] for i in vi]).to(DEVICE)
        vA  = torch.stack([train_valid[i][1] for i in vi]).to(DEVICE)
        vZ  = torch.stack([train_valid[i][2] for i in vi]).to(DEVICE)
        cz0 = torch.stack([corr[i][0] for i in ci]).to(DEVICE)
        cA  = torch.stack([corr[i][1] for i in ci]).to(DEVICE)
        cZ  = torch.stack([corr[i][2] for i in ci]).to(DEVICE)
        Zb = torch.cat([vZ, cZ], dim=0)
        z0b = torch.cat([vz0, cz0], dim=0)
        Ab = torch.cat([vA, cA], dim=0)
        with torch.no_grad():
            targets = torch.cat([
                torch.zeros(len(vi), device=DEVICE),
                torch.sigmoid(teacher(cZ)).to(DEVICE)
            ])
        opt.zero_grad()
        energies = model(z0b, Ab, Zb)
        loss = F.mse_loss(energies, targets * 5.0 - 1.0) + 0.01 * energies.pow(2).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
    return model.eval()

# E: Curriculum
def train_curriculum(seed):
    set_seed(seed)
    model = FusedIWCMEnergy(d_slot=D_SLOT, d_action=D_ACTION, hidden=128,
                            num_slots=N_SLOTS).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    nv = len(train_valid)
    # Phase 1: crude
    crude_corr = []
    for z0, A, Z in train_valid:
        Zc = Z.clone() + torch.randn_like(Z) * 0.3
        Zc[:, :, :5] = Z[:, :, :5]; Zc[:, :, 15:] = Z[:, :, 15:]
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
    # Phase 2: mixed
    for ep in range(50, 100):
        corr = generate_mixed(train_valid)
        nc = len(corr)
        vi = np.random.choice(nv, min(BATCH, nv), replace=False)
        ci = np.random.choice(nc, min(BATCH, nc), replace=False)
        vz0 = torch.stack([train_valid[i][0] for i in vi]).to(DEVICE)
        vA  = torch.stack([train_valid[i][1] for i in vi]).to(DEVICE)
        vZ  = torch.stack([train_valid[i][2] for i in vi]).to(DEVICE)
        cz0 = torch.stack([corr[i][0] for i in ci]).to(DEVICE)
        cA  = torch.stack([corr[i][1] for i in ci]).to(DEVICE)
        cZ  = torch.stack([corr[i][2] for i in ci]).to(DEVICE)
        opt.zero_grad()
        ev = model(vz0, vA, vZ); ec = model(cz0, cA, cZ)
        loss = F.relu(ev + 1.0).mean() + F.relu(1.0 - ec).mean() + \
               0.001*(ev.pow(2).mean() + ec.pow(2).mean())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
    # Phase 3: high-teacher
    corr = generate_mixed(train_valid)
    t_scores = []
    for _, _, Z in corr:
        t_scores.append(torch.sigmoid(teacher(Z.unsqueeze(0).to(DEVICE))).item())
    high_corr = [corr[i] for i in np.where(np.array(t_scores) > np.median(t_scores))[0]]
    for ep in range(100, 150):
        nc = len(high_corr)
        vi = np.random.choice(nv, min(BATCH, nv), replace=False)
        ci = np.random.choice(nc, min(BATCH, nc), replace=False)
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

# ═══════════════════════════════════════════════════════════════════════════════
# Run all
# ═══════════════════════════════════════════════════════════════════════════════

run_strat("A: Oracle           ", train_oracle)
run_strat("B: Raw pseudo       ", train_raw)
run_strat("C: Teacher-weighted ", train_weighted)
run_strat("D: Teacher regress  ", train_regression)
run_strat("E: Curriculum       ", train_curriculum)
run_strat("F: Adversarial TAMG ", train_tamg_v2)

# ═══════════════════════════════════════════════════════════════════════════════
# Verdict
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 75)
print("TAMG-v2 VERDICT")
print("=" * 75)

print(f"\n{'Strategy':<28} {'OVERALL':>10}")
print("-" * 40)
for name in results_all:
    mean, std = results_all[name]["OVERALL"]
    print(f"  {name:<26} {mean:8.4f} ± {std:.3f}")

oracle_mean = results_all["A: Oracle           "]["OVERALL"][0]
adv_data = results_all.get("F: Adversarial TAMG ")
adv_mean = adv_data["OVERALL"][0] if adv_data else 0.0
raw_mean = results_all["B: Raw pseudo       "]["OVERALL"][0]

print(f"\n  Oracle:                     {oracle_mean:.4f}")
print(f"  Raw pseudo:                 {raw_mean:.4f}")
print(f"  Adversarial TAMG:           {adv_mean:.4f}")
print(f"  Gap (adv → oracle):         {oracle_mean - adv_mean:.4f}")
print(f"  Gain over raw:              {adv_mean - raw_mean:+.4f}")
print(f"  Teacher AUROC:              {t_auroc:.4f}")

if adv_mean >= 0.85:
    print(f"\n  TAMG-v2 PASSES: Adversarial mining closes the gap.")
    print(f"  TAMG is salvageable as teacher-guided counterexample mining.")
elif adv_mean >= 0.80:
    print(f"\n  MARGINAL: Modest improvement, below 0.85 threshold.")
    print(f"  Adversarial mining helps but doesn't close the gap.")
else:
    print(f"\n  HARD STOP: Adversarial mining does not salvage TAMG.")
    print(f"  Teacher-guided counterexample search cannot train IWCM.")
    print(f"  TAMG is not viable as an IWCM training method.")
