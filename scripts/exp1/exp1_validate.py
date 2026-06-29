#!/usr/bin/env python3
"""Comprehensive Experiment 1 validation: 5-seed significance, per-head
breakdown, longer horizons, and FusedIWCMEnergy ablation."""
import sys, pickle, numpy as np, torch, torch.nn.functional as F
from pathlib import Path
from collections import defaultdict
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.seed import set_seed
from src.iwcm.slot_energy import SlotIWCMEnergy
from src.iwcm.fused_energy import FusedIWCMEnergy
from src.env.oracle_slot_encoder import ORACLE_SLOT_DIM, MAX_OBJECTS
from sklearn.metrics import roc_auc_score

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ─── Load data ────────────────────────────────────────────────────────────────
with open("data/compositional_grid.pkl", "rb") as f:
    data = pickle.load(f)

train_valid = [(torch.from_numpy(z0).float(), torch.from_numpy(A).float(),
                torch.from_numpy(Z).float()) for z0, A, Z in data["train_valid"]]
train_corr_raw = [(torch.from_numpy(item[0][0]).float(), torch.from_numpy(item[0][1]).float(),
                   torch.from_numpy(item[0][2]).float(), item[1]) for item in data["train_corr"]]
test_valid = [(torch.from_numpy(z0).float(), torch.from_numpy(A).float(),
               torch.from_numpy(Z).float()) for z0, A, Z in data["test_valid"]]
test_corr_raw = [(torch.from_numpy(item[0][0]).float(), torch.from_numpy(item[0][1]).float(),
                  torch.from_numpy(item[0][2]).float(), item[1]) for item in data["test_corr"]]

N, d, H = MAX_OBJECTS, ORACLE_SLOT_DIM, 25
train_corr_struct = [(z0, A, Z) for z0, A, Z, _ in train_corr_raw]
n_corr = len(train_corr_struct)

test_by_type = defaultdict(list)
for z0, A, Z, meta in test_corr_raw:
    test_by_type[meta["violation_type"]].append((z0, A, Z))
vtypes = sorted(test_by_type.keys())

def train_eval(seed, model_class, model_kwargs, train_corr_data, epochs, label):
    set_seed(seed)
    model = model_class(**model_kwargs).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    batch = 32
    for ep in range(epochs):
        vi = np.random.choice(len(train_valid), min(batch, len(train_valid)), replace=False)
        ci = np.random.choice(len(train_corr_data), min(batch, len(train_corr_data)), replace=False)
        vz0 = torch.stack([train_valid[i][0] for i in vi]).to(DEVICE)
        vA = torch.stack([train_valid[i][1] for i in vi]).to(DEVICE)
        vZ = torch.stack([train_valid[i][2] for i in vi]).to(DEVICE)
        cz0 = torch.stack([train_corr_data[i][0] for i in ci]).to(DEVICE)
        cA = torch.stack([train_corr_data[i][1] for i in ci]).to(DEVICE)
        cZ = torch.stack([train_corr_data[i][2] for i in ci]).to(DEVICE)
        opt.zero_grad()
        ev = model(vz0, vA, vZ); ec = model(cz0, cA, cZ)
        loss = F.relu(ev + 1.0).mean() + F.relu(1.0 - ec).mean() + 0.001*(ev.pow(2).mean()+ec.pow(2).mean())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
    model.eval()
    with torch.no_grad():
        valid_e = []
        for z0, A, Z in test_valid:
            valid_e.append(model(z0.unsqueeze(0).to(DEVICE), A.unsqueeze(0).to(DEVICE),
                                 Z.unsqueeze(0).to(DEVICE)).item())
        per_type = {}
        for vt in vtypes:
            corr_e = []
            for z0, A, Z in test_by_type[vt]:
                corr_e.append(model(z0.unsqueeze(0).to(DEVICE), A.unsqueeze(0).to(DEVICE),
                                    Z.unsqueeze(0).to(DEVICE)).item())
            labels = [0]*len(valid_e) + [1]*len(corr_e)
            energies = valid_e + corr_e
            per_type[vt] = {'auroc': roc_auc_score(labels, energies),
                            'ev': np.mean(valid_e), 'ei': np.mean(corr_e)}
    return per_type, model

# ═══════════════════════════════════════════════════════════════════════════════
# 1. 5-SEED SIGNIFICANCE
# ═══════════════════════════════════════════════════════════════════════════════
print("="*70)
print("1. 5-SEED SIGNIFICANCE: Model C (compositional)")
print("="*70)

all_seeds = {}
for seed in [42, 123, 456, 789, 1024]:
    print(f"\n  Seed {seed}...")
    np.random.seed(seed)
    # Generate random corruptions for this seed's Model B baseline
    random_corr = []
    for i in range(n_corr):
        idx = np.random.randint(0, len(train_valid))
        z0, A, Z = train_valid[idx]
        Z_noisy = Z.clone() + torch.randn_like(Z) * 0.3
        Z_noisy[:, :, :5] = Z[:, :, :5]
        Z_noisy[:, :, 15:] = Z[:, :, 15:]
        random_corr.append((z0.clone(), A.clone(), Z_noisy))

    rb, _ = train_eval(seed, SlotIWCMEnergy,
                        dict(d_slot=d, d_action=11, hidden_dim=128, num_slots=N),
                        random_corr, 150, f"B-{seed}")
    rc, _ = train_eval(seed, SlotIWCMEnergy,
                        dict(d_slot=d, d_action=11, hidden_dim=128, num_slots=N),
                        train_corr_struct, 150, f"C-{seed}")

    all_seeds[seed] = {'B': rb, 'C': rc}

print(f"\n{'Type':<14} ", end="")
for seed in all_seeds:
    print(f"{'B'+str(seed)[-2:]:>8} {'C'+str(seed)[-2:]:>8} {'Δ':>6}", end="  ")
print(f"{'B_mean':>8} {'C_mean':>8} {'Δ_mean':>8} {'±':>8}")
print("-"*140)

for vt in vtypes:
    b_vals = [all_seeds[s]['B'][vt]['auroc'] for s in all_seeds]
    c_vals = [all_seeds[s]['C'][vt]['auroc'] for s in all_seeds]
    deltas = [c - b for c, b in zip(c_vals, b_vals)]
    print(f"{vt:<14} ", end="")
    for b, c in zip(b_vals, c_vals):
        print(f"{b:8.3f} {c:8.3f} {c-b:+6.3f}", end="  ")
    print(f"{np.mean(b_vals):8.3f} {np.mean(c_vals):8.3f} {np.mean(deltas):+8.3f} {np.std(deltas):8.3f}")

all_b = [np.mean([all_seeds[s]['B'][vt]['auroc'] for vt in vtypes]) for s in all_seeds]
all_c = [np.mean([all_seeds[s]['C'][vt]['auroc'] for vt in vtypes]) for s in all_seeds]
all_d = [c - b for c, b in zip(all_c, all_b)]
print(f"\nOVERALL: B={np.mean(all_b):.3f}±{np.std(all_b):.3f}  C={np.mean(all_c):.3f}±{np.std(all_c):.3f}  Δ={np.mean(all_d):+.3f}±{np.std(all_d):.3f}")

# ═══════════════════════════════════════════════════════════════════════════════
# 2. PER-HEAD BREAKDOWN
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("2. PER-HEAD ENERGY BREAKDOWN")
print("="*70)

set_seed(42)
_, model_c_head = train_eval(42, SlotIWCMEnergy,
                         dict(d_slot=d, d_action=11, hidden_dim=128, num_slots=N),
                         train_corr_struct, 150, "C-analysis")

with torch.no_grad():
    print(f"\n{'Head':<16} {'E_valid_mean':>14} {'E_invalid_mean':>14} {'Margin':>10} {'AUROC':>10}")
    print("-"*70)
    for head_name in ['boundary', 'local', 'invariant', 'effect', 'counterfactual']:
        # Per-head energies for valid
        valid_head_e = []
        for z0, A, Z in test_valid:
            per_head = model_c_head.per_head(z0.unsqueeze(0).to(DEVICE),
                                                 A.unsqueeze(0).to(DEVICE),
                                                 Z.unsqueeze(0).to(DEVICE))
            valid_head_e.append(per_head[head_name].item())

        # Per-head energies for ALL corrupt types combined
        all_corr = []
        for vt in vtypes:
            for z0, A, Z in test_by_type[vt]:
                all_corr.append((z0, A, Z))

        corr_head_e = []
        for z0, A, Z in all_corr:
            per_head = model_c_head.per_head(z0.unsqueeze(0).to(DEVICE),
                                                 A.unsqueeze(0).to(DEVICE),
                                                 Z.unsqueeze(0).to(DEVICE))
            corr_head_e.append(per_head[head_name].item())

        labels = [0]*len(valid_head_e) + [1]*len(corr_head_e)
        energies = valid_head_e + corr_head_e
        auroc = roc_auc_score(labels, energies)
        print(f"{head_name:<16} {np.mean(valid_head_e):14.3f} {np.mean(corr_head_e):14.3f} "
              f"{np.mean(corr_head_e)-np.mean(valid_head_e):10.3f} {auroc:10.3f}")

    # Per-head per-type breakdown
    print(f"\n{'Type':<14} ", end="")
    heads = ['boundary', 'local', 'invariant', 'effect', 'counterfactual']
    for h in heads:
        print(f"{h[:6]:>8}", end=" ")
    print()
    print("-"*65)

    for vt in vtypes:
        print(f"{vt:<14} ", end="")
        for head_name in heads:
            corr_e = []
            for z0, A, Z in test_by_type[vt]:
                ph = model_c_head.per_head(z0.unsqueeze(0).to(DEVICE),
                                               A.unsqueeze(0).to(DEVICE),
                                               Z.unsqueeze(0).to(DEVICE))
                corr_e.append(ph[head_name].item())
            labels = [0]*len(valid_head_e) + [1]*len(corr_e)
            energies = valid_head_e + corr_e
            auroc = roc_auc_score(labels, energies)
            print(f"{auroc:8.3f}", end=" ")
        print()

# ═══════════════════════════════════════════════════════════════════════════════
# 3. FUSED IWCM ABLATION
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("3. ABLATION: FusedIWCMEnergy (52K) vs SlotIWCMEnergy (579K)")
print("="*70)

set_seed(42)
_, fused_model = train_eval(42, FusedIWCMEnergy,
                       dict(d_slot=d, d_action=11, hidden=128, num_slots=N),
                       train_corr_struct, 150, "Fused")

with torch.no_grad():
    f_valid = []
    for z0, A, Z in test_valid:
        f_valid.append(fused_model(z0.unsqueeze(0).to(DEVICE), A.unsqueeze(0).to(DEVICE),
                             Z.unsqueeze(0).to(DEVICE)).item())

print(f"\n{'Type':<14} {'Fused(52K)':>12} {'Slot(579K)':>12} {'Δ':>10}")
print("-"*55)
fused_aurocs = []
for vt in vtypes:
    f_corr = []
    for z0, A, Z in test_by_type[vt]:
        f_corr.append(fused_model(z0.unsqueeze(0).to(DEVICE), A.unsqueeze(0).to(DEVICE),
                            Z.unsqueeze(0).to(DEVICE)).item())
    fa = roc_auc_score([0]*len(f_valid)+[1]*len(f_corr), f_valid+f_corr)
    sa = all_seeds[42]['C'][vt]['auroc']
    fused_aurocs.append(fa)
    print(f"{vt:<14} {fa:12.3f} {sa:12.3f} {fa-sa:+10.3f}")

fa_avg = np.mean(fused_aurocs)
sa_avg = np.mean([all_seeds[42]['C'][vt]['auroc'] for vt in vtypes])
print(f"{'AVERAGE':<14} {fa_avg:12.3f} {sa_avg:12.3f} {fa_avg-sa_avg:+10.3f}")
print(f"\nFusedIWCMEnergy params: 52K, SlotIWCMEnergy params: 579K")
print(f"Fused is {579/52:.0f}x smaller. Delta: {fa_avg-sa_avg:+.3f}")

print("\n" + "="*70)
print("ALL VALIDATION COMPLETE")
