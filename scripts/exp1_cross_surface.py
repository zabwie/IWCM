#!/usr/bin/env python3
"""Experiment 1: Cross-Surface Law Generalization on Compositional Grid.

The compositional corruption grid has 5 independent axes (object type, context,
violation type, time gap, distractors) with compositional train/test split.
Tests whether the model learns causal laws or surface patterns.

Uses oracle slot format (19-dim per object) with SlotIWCMEnergy (5 heads).
"""
import sys, pickle, numpy as np, torch, torch.nn.functional as F
from pathlib import Path
from collections import defaultdict
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.seed import set_seed
from src.iwcm.slot_energy import SlotIWCMEnergy
from src.encoder.oracle_slot_encoder import ORACLE_SLOT_DIM, MAX_OBJECTS
from sklearn.metrics import roc_auc_score

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
set_seed(42)

with open("data/compositional_grid.pkl", "rb") as f:
    data = pickle.load(f)

train_valid = [(torch.from_numpy(z0).float(), torch.from_numpy(A).float(),
                torch.from_numpy(Z).float())
               for z0, A, Z in data["train_valid"]]
train_corr = [(torch.from_numpy(item[0][0]).float(), torch.from_numpy(item[0][1]).float(),
               torch.from_numpy(item[0][2]).float(), item[1])
              for item in data["train_corr"]]
test_valid = [(torch.from_numpy(z0).float(), torch.from_numpy(A).float(),
               torch.from_numpy(Z).float())
              for z0, A, Z in data["test_valid"]]
test_corr = [(torch.from_numpy(item[0][0]).float(), torch.from_numpy(item[0][1]).float(),
              torch.from_numpy(item[0][2]).float(), item[1])
             for item in data["test_corr"]]

N, d, H = MAX_OBJECTS, ORACLE_SLOT_DIM, 25
print(f"Train: {len(train_valid)} valid, {len(train_corr)} corrupt")
print(f"Test:  {len(test_valid)} valid, {len(test_corr)} corrupt")

train_by_type = defaultdict(list)
for z0, A, Z, meta in train_corr:
    train_by_type[meta["violation_type"]].append((z0, A, Z))
test_by_type = defaultdict(list)
for z0, A, Z, meta in test_corr:
    test_by_type[meta["violation_type"]].append((z0, A, Z))

print(f"Violation types: {sorted(train_by_type.keys())}")
for vt in sorted(train_by_type.keys()):
    print(f"  {vt}: train={len(train_by_type[vt])} test={len(test_by_type[vt])}")

print(f"\nTraining SlotIWCMEnergy (5 heads)...")
model = SlotIWCMEnergy(d_slot=d, d_action=11, hidden_dim=128, num_slots=N).to(DEVICE)
opt = torch.optim.Adam(model.parameters(), lr=3e-3)

batch = 32
for ep in range(150):
    vi = np.random.choice(len(train_valid), min(batch, len(train_valid)), replace=False)
    ci = np.random.choice(len(train_corr), min(batch, len(train_corr)), replace=False)
    vz0 = torch.stack([train_valid[i][0] for i in vi]).to(DEVICE)
    vA = torch.stack([train_valid[i][1] for i in vi]).to(DEVICE)
    vZ = torch.stack([train_valid[i][2] for i in vi]).to(DEVICE)
    cz0 = torch.stack([train_corr[i][0] for i in ci]).to(DEVICE)
    cA = torch.stack([train_corr[i][1] for i in ci]).to(DEVICE)
    cZ = torch.stack([train_corr[i][2] for i in ci]).to(DEVICE)
    opt.zero_grad()
    ev = model(vz0, vA, vZ)
    ec = model(cz0, cA, cZ)
    loss = F.relu(ev + 1.0).mean() + F.relu(1.0 - ec).mean() + \
           0.001 * (ev.pow(2).mean() + ec.pow(2).mean())
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    if (ep + 1) % 50 == 0:
        print(f"  ep {ep+1:3d}: loss={loss.item():.4f} ev={ev.mean().item():+.2f} ec={ec.mean().item():+.2f}")

model.eval()
print("\n" + "=" * 60)
print("Per-Violation-Type AUROC (Cross-Surface)")
print(f"{'Type':<14} {'AUROC':>8} {'E_valid':>10} {'E_invalid':>10} {'Margin':>10}")
print("-" * 60)

all_energies, all_labels = [], []
with torch.no_grad():
    for z0, A, Z in test_valid:
        e = model(z0.unsqueeze(0).to(DEVICE), A.unsqueeze(0).to(DEVICE),
                  Z.unsqueeze(0).to(DEVICE)).item()
        all_energies.append(e); all_labels.append(0)

    per_type_results = {}
    for vt in sorted(test_by_type.keys()):
        energies = []
        for z0, A, Z in test_by_type[vt]:
            e = model(z0.unsqueeze(0).to(DEVICE), A.unsqueeze(0).to(DEVICE),
                      Z.unsqueeze(0).to(DEVICE)).item()
            energies.append(e)
        vt_labels = [0]*len(all_energies) + [1]*len(energies)
        vt_energies = all_energies + energies
        auroc = roc_auc_score(vt_labels, vt_energies)
        ev_m = np.mean(all_energies)
        ei_m = np.mean(energies)
        print(f"{vt:<14} {auroc:8.3f} {ev_m:10.2f} {ei_m:10.2f} {ei_m-ev_m:10.2f}")
        per_type_results[vt] = auroc
        all_energies.extend(energies); all_labels.extend([1]*len(energies))

overall = roc_auc_score(all_labels, all_energies)
ev_all = np.mean([e for e,l in zip(all_energies,all_labels) if l==0])
ei_all = np.mean([e for e,l in zip(all_energies,all_labels) if l==1])
print("-" * 60)
print(f"{'OVERALL':<14} {overall:8.3f} {ev_all:10.2f} {ei_all:10.2f} {ei_all-ev_all:10.2f}")

print("\nCross-Surface Verdict:")
high_types = [vt for vt, a in per_type_results.items() if a >= 0.75]
low_types = [vt for vt, a in per_type_results.items() if a < 0.65]
print(f"  High AUROC (>0.75): {high_types if high_types else 'none'}")
print(f"  Low AUROC (<0.65):  {low_types if low_types else 'none'}")

if overall >= 0.85:
    print(f"\nSTRONG — {overall:.3f} overall AUROC across compositional split.")
    print("  Model learns causal laws, not surface patterns.")
elif overall >= 0.75:
    print(f"\nPROMISING — {overall:.3f} overall. Some laws generalize.")
elif overall >= 0.65:
    print(f"\nMODERATE — {overall:.3f} overall. Partial signal.")
else:
    print(f"\nWEAK — {overall:.3f} overall.")
