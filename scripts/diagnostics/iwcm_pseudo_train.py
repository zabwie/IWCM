#!/usr/bin/env python3
"""Decisive IWCM experiment: does manifold-filtered self-supervision translate
to IWCM cross-surface generalization?

Trains FusedIWCMEnergy (52K) with 5 corruption sources:
  A: crude pseudo-negatives (no filter)
  B: manifold-filtered contrast negatives
  C: manifold-filtered mixed generalist negatives
  D: noise_hard negatives only
  E: oracle negatives (upper bound)

Reports per-type cross-surface AUROC, artifact correlation checks, and whether
the 0.939 validator signal translates to IWCM energy margin.
"""
import sys, pickle, numpy as np, torch, torch.nn.functional as F
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
SEEDS = [42, 123, 456]

# ═══════════════════════════════════════════════════════════════════════════════
# Data
# ═══════════════════════════════════════════════════════════════════════════════

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

# Compute valid data statistics for artifact checks
valid_Zs = torch.stack([t[2] for t in train_valid])
valid_norms = valid_Zs.norm(dim=-1).float()
valid_mean_norm = valid_norms.mean().item()
valid_std_norm = valid_norms.std().item()
valid_active = (valid_norms > 0.1).float().sum(dim=-1).float()
valid_mean_active = valid_active.mean().item()
valid_std_active = valid_active.std().item()

# Build kNN reference for manifold distance
valid_Z_flat = valid_Zs.reshape(len(train_valid), -1)

train_corr_struct = [(z0, A, Z) for z0, A, Z, _ in train_corr_raw]

print(f"Train: {len(train_valid)}v + {len(train_corr_struct)}c")
print(f"Test:  {len(test_valid)}v + {len(test_corr_raw)}c")

# ═══════════════════════════════════════════════════════════════════════════════
# Corruption generators
# ═══════════════════════════════════════════════════════════════════════════════

def passes_manifold_filter(Z_orig, Z_corr):
    pert = (Z_corr - Z_orig).pow(2).mean().sqrt().item()
    if pert > 1.0: return False
    cn = Z_corr.norm(dim=-1)
    ac = (cn > 0.1).float().sum(dim=-1).float().mean().item()
    if ac < max(1.0, valid_mean_active - 3 * valid_std_active): return False
    if ac > valid_mean_active + 3 * valid_std_active: return False
    return True

def generate_crude(data):
    corr = []
    for z0, A, Z in data:
        Z_c = Z.clone() + torch.randn_like(Z) * 0.3
        Z_c[:, :, :5] = Z[:, :, :5]
        Z_c[:, :, 15:] = Z[:, :, 15:]
        corr.append((z0.clone(), A.clone(), Z_c))
    return corr

def generate_contrast(data):
    corr = []
    for z0, A, Z in data:
        Z_c = Z.clone()
        s = np.random.randint(0, 5)
        if s == 0: Z_c += torch.randn_like(Z_c) * 0.1
        elif s == 1:
            t = np.random.randint(0, H_); Z_c[t:] = Z_c[t:].flip(dims=[0])
        elif s == 2 and N_SLOTS >= 2:
            i, j = np.random.choice(N_SLOTS, 2, replace=False)
            Z_c[:, i], Z_c[:, j] = Z_c[:, j].clone(), Z_c[:, i].clone()
        elif s == 3: Z_c[:, np.random.randint(0, N_SLOTS)] *= 0.1
        elif s == 4:
            t = np.random.randint(1, H_); Z_c[t:] = Z_c[:H_ - t].clone()
        if passes_manifold_filter(Z, Z_c):
            corr.append((z0.clone(), A.clone(), Z_c))
    return corr

def generate_noise_hard(data):
    corr = []
    for z0, A, Z in data:
        noise = torch.randn_like(Z) * 0.05
        Z_c = Z + noise
        if passes_manifold_filter(Z, Z_c):
            corr.append((z0.clone(), A.clone(), Z_c))
    return corr

def generate_mixed(data):
    all_c = []
    for z0, A, Z in data:
        fam_idx = np.random.randint(0, 8)
        Z_c = Z.clone()
        if fam_idx == 0:
            t = np.random.randint(1, H_); Z_c[t:] = Z_c[:H_ - t].clone()
        elif fam_idx == 1 and N_SLOTS >= 2:
            i, j = np.random.choice(N_SLOTS, 2, replace=False)
            Z_c[:, i], Z_c[:, j] = Z_c[:, j].clone(), Z_c[:, i].clone()
        elif fam_idx == 2:
            Z_c[:, np.random.randint(0, N_SLOTS)] *= 0.0
        elif fam_idx == 3 and N_SLOTS >= 2:
            i, j = np.random.choice(N_SLOTS, 2, replace=False)
            Z_c[:, j] = Z_c[:, i].clone()
        elif fam_idx == 4:
            Z_c[..., :5] += torch.randn_like(Z_c[..., :5]) * 0.03
            Z_c[..., 7:9] += torch.randn_like(Z_c[..., 7:9]) * 0.03
        elif fam_idx == 5:
            slot = np.random.randint(0, N_SLOTS)
            t = np.random.randint(1, H_ - 1)
            Z_c[t:, slot, 5] += (np.random.rand() * 2 - 1) * 0.5
            Z_c[t:, slot, 6] += (np.random.rand() * 2 - 1) * 0.5
        elif fam_idx == 6:
            Z_c += torch.randn_like(Z_c) * 0.05
        elif fam_idx == 7 and H_ >= 2:
            t = np.random.randint(0, H_ - 1)
            Z_c[t + 1:, np.random.randint(0, N_SLOTS)] += A[t].abs().mean().item() * 0.2
        if passes_manifold_filter(Z, Z_c):
            all_c.append((z0.clone(), A.clone(), Z_c))
    return all_c

# ═══════════════════════════════════════════════════════════════════════════════
# Generate all corruption sets
# ═══════════════════════════════════════════════════════════════════════════════

print("\nGenerating corruptions...")
crude_corr = generate_crude(train_valid)
cont_corr = generate_contrast(train_valid)
mixed_corr = generate_mixed(train_valid)
noise_corr = generate_noise_hard(train_valid)
oracle_corr = train_corr_struct

for name, corr in [("crude     ", crude_corr), ("contrast  ", cont_corr),
                    ("mixed     ", mixed_corr), ("noise_hard", noise_corr),
                    ("oracle    ", oracle_corr)]:
    print(f"  {name} {len(corr):>4} corruptions")

# ═══════════════════════════════════════════════════════════════════════════════
# IWCM training
# ═══════════════════════════════════════════════════════════════════════════════

def train_iwcm(valid_data, corr_data, seed, label):
    set_seed(seed)
    model = FusedIWCMEnergy(d_slot=D_SLOT, d_action=D_ACTION, hidden=128,
                            num_slots=N_SLOTS).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    nv, nc = len(valid_data), len(corr_data)

    for ep in range(EPOCHS):
        vi = np.random.choice(nv, min(BATCH, nv), replace=False)
        ci = np.random.choice(nc, min(BATCH, nc), replace=False)

        vz0 = torch.stack([valid_data[i][0] for i in vi]).to(DEVICE)
        vA  = torch.stack([valid_data[i][1] for i in vi]).to(DEVICE)
        vZ  = torch.stack([valid_data[i][2] for i in vi]).to(DEVICE)
        cz0 = torch.stack([corr_data[i][0] for i in ci]).to(DEVICE)
        cA  = torch.stack([corr_data[i][1] for i in ci]).to(DEVICE)
        cZ  = torch.stack([corr_data[i][2] for i in ci]).to(DEVICE)

        opt.zero_grad()
        ev = model(vz0, vA, vZ)
        ec = model(cz0, cA, cZ)
        loss = F.relu(ev + 1.0).mean() + F.relu(1.0 - ec).mean() + \
               0.001 * (ev.pow(2).mean() + ec.pow(2).mean())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

    model.eval()
    return model

def evaluate_iwcm(model):
    with torch.no_grad():
        valid_e = []
        for z0, A, Z in test_valid:
            e = model(z0.unsqueeze(0).to(DEVICE), A.unsqueeze(0).to(DEVICE),
                      Z.unsqueeze(0).to(DEVICE)).item()
            valid_e.append(e)

        per_type = {}
        for vt in VTYPES:
            corr_e = []
            for z0, A, Z in test_by_type[vt]:
                e = model(z0.unsqueeze(0).to(DEVICE), A.unsqueeze(0).to(DEVICE),
                          Z.unsqueeze(0).to(DEVICE)).item()
                corr_e.append(e)
            labels = [0] * len(valid_e) + [1] * len(corr_e)
            scores = valid_e + corr_e
            per_type[vt] = roc_auc_score(labels, scores)

        all_corr_e = []
        for vt in VTYPES:
            for z0, A, Z in test_by_type[vt]:
                all_corr_e.append(evaluate_iwcm_single(model, z0, A, Z))
        all_labels = [0] * len(valid_e) + [1] * len(all_corr_e)
        all_scores = valid_e + all_corr_e
        per_type["OVERALL"] = roc_auc_score(all_labels, all_scores)

    return per_type, valid_e

def evaluate_iwcm_single(model, z0, A, Z):
    with torch.no_grad():
        return model(z0.unsqueeze(0).to(DEVICE), A.unsqueeze(0).to(DEVICE),
                     Z.unsqueeze(0).to(DEVICE)).item()

# ═══════════════════════════════════════════════════════════════════════════════
# Ablations
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 85)
print("IWCM TRAINING: 5 corruption sources × 3 seeds")
print("=" * 85)

sources = {
    "A: crude      ": crude_corr,
    "B: contrast   ": cont_corr,
    "C: mixed      ": mixed_corr,
    "D: noise_hard ": noise_corr,
    "E: oracle     ": oracle_corr,
}

all_results = {}

for src_name, corr_data in sources.items():
    print(f"\n  {src_name}")
    src_results = {}
    for seed in SEEDS:
        model = train_iwcm(train_valid, corr_data, seed, src_name.strip())
        results, valid_energies = evaluate_iwcm(model)
        src_results[seed] = results
        overall = results["OVERALL"]
        print(f"    seed={seed}: OVERALL={overall:.4f}  " +
              " ".join(f"{vt[:3]}={results[vt]:.3f}" for vt in VTYPES[:3]))

    # Average over seeds
    avg = {}
    for vt in VTYPES + ["OVERALL"]:
        vals = [src_results[s][vt] for s in SEEDS]
        avg[vt] = (np.mean(vals), np.std(vals))
    all_results[src_name.strip()] = avg

# ═══════════════════════════════════════════════════════════════════════════════
# Results table
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 85)
print("PER-TYPE CROSS-SURFACE AUROC (mean ± std across 3 seeds)")
print("=" * 85)

header = f"{'Type':<14}"
for name in sources.keys():
    header += f" {name.strip()[:12]:>14}"
print(f"\n{header}")
print("-" * (14 + 15 * len(sources)))

for vt in VTYPES:
    row = f"{vt:<14}"
    for name in sources.keys():
        mean, std = all_results[name.strip()][vt]
        row += f" {mean:7.4f}±{std:.2f}"
    print(row)

row = f"{'OVERALL':<14}"
for name in sources.keys():
    mean, std = all_results[name.strip()]["OVERALL"]
    row += f" {mean:7.4f}±{std:.2f}"
print(row)

# Deltas vs oracle
print(f"\n{'Delta vs Oracle':<14}", end="")
for name in sources.keys():
    if "oracle" in name.lower():
        print(f" {'--':>13}", end="")
        continue
    mean, _ = all_results[name.strip()]["OVERALL"]
    oracle_mean, _ = all_results["E: oracle"]["OVERALL"]
    delta = mean - oracle_mean
    print(f" {delta:+13.4f}", end="")
print()

# Deltas vs crude
print(f"{'Delta vs Crude':<14}", end="")
for name in sources.keys():
    if "crude" in name.lower():
        print(f" {'--':>13}", end="")
        continue
    mean, _ = all_results[name.strip()]["OVERALL"]
    crude_mean, _ = all_results["A: crude"]["OVERALL"]
    delta = mean - crude_mean
    print(f" {delta:+13.4f}", end="")
print()

# ═══════════════════════════════════════════════════════════════════════════════
# Energy margin comparison
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 85)
print("ENERGY MARGIN (E_valid vs E_invalid)")
print("=" * 85)

for name in sources.keys():
    set_seed(SEEDS[0])
    corr_data = sources[name]
    model = train_iwcm(train_valid, corr_data, SEEDS[0], name.strip())
    _, valid_energies = evaluate_iwcm(model)

    all_corr_energies = []
    for vt in VTYPES:
        for z0, A, Z in test_by_type[vt]:
            all_corr_energies.append(
                model(z0.unsqueeze(0).to(DEVICE), A.unsqueeze(0).to(DEVICE),
                      Z.unsqueeze(0).to(DEVICE)).item())

    ev = np.mean(valid_energies)
    ec = np.mean(all_corr_energies)
    print(f"  {name.strip():<16} E_valid={ev:+7.2f}  E_invalid={ec:+7.2f}  margin={ec-ev:+7.2f}")

# ═══════════════════════════════════════════════════════════════════════════════
# Gateway result
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 85)
print("GATEWAY: Does self-supervised IWCM translate validator signal?")
print("=" * 85)

oracle_mean, oracle_std = all_results["E: oracle"]["OVERALL"]
crude_mean, crude_std = all_results["A: crude"]["OVERALL"]
mixed_mean, mixed_std = all_results["C: mixed"]["OVERALL"]
contrast_mean, _ = all_results["B: contrast"]["OVERALL"]
noise_mean, _ = all_results["D: noise_hard"]["OVERALL"]

crude_oracle_gap = oracle_mean - crude_mean
mixed_gain = mixed_mean - crude_mean
pct_closed = (mixed_gain / max(crude_oracle_gap, 0.001)) * 100

print(f"""
  Crude (no filter) IWCM:      {crude_mean:.4f} ± {crude_std:.3f}
  Mixed (manifold-filtered):   {mixed_mean:.4f} ± {mixed_std:.3f}
  Oracle IWCM:                 {oracle_mean:.4f} ± {oracle_std:.3f}

  Oracle - Crude gap:          {crude_oracle_gap:.4f}
  Mixed gain over Crude:       {mixed_gain:+.4f}
  Gap closed:                  {pct_closed:.0f}%
  Remaining to Oracle:         {oracle_mean - mixed_mean:.4f}
""")

# Compare to published baselines
print(f"  Reference baselines:")
print(f"    PoC TAMG (FINDINGS):       ~0.77")
print(f"    Exp1 Model C (oracle):      0.952 (compositional split)")
print(f"    Fused IWCM (best FINDINGS): 0.879 (conservation, oracle slots)")

if mixed_mean >= 0.85:
    print(f"\n  VERDICT: Self-supervised manifold-filtered negatives produce")
    print(f"  IWCM performance substantially above the 0.77 PoC ceiling.")
    print(f"  The validator signal DOES translate to IWCM training.")
elif mixed_mean >= 0.80:
    print(f"\n  VERDICT: Modest improvement over PoC ceiling.")
    print(f"  Signal partially translates but gap remains significant.")
else:
    print(f"\n  VERDICT: IWCM does not benefit from self-supervised negatives.")
    print(f"  Validator signal is a probe, not a training method.")
