#!/usr/bin/env python3
"""Experiment 1: Model B (random corruptions) vs Model C (compositional grid).

The paper's critical ablation: does the compositional corruption curriculum (C)
produce better cross-surface generalization than random corruptions (B)?
"""
import sys, pickle, numpy as np, torch, torch.nn.functional as F
from pathlib import Path
from collections import defaultdict
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.seed import set_seed
from src.iwcm.slot_energy import SlotIWCMEnergy
from src.env.oracle_slot_encoder import ORACLE_SLOT_DIM, MAX_OBJECTS
from sklearn.metrics import roc_auc_score

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def train_model(train_valid, train_corr, epochs, label, d_slot, d_action, N):
    """Train SlotIWCMEnergy with 5 decomposed heads."""
    set_seed(42)
    model = SlotIWCMEnergy(d_slot=d_slot, d_action=d_action, hidden_dim=128, num_slots=N).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    batch = 32

    for ep in range(epochs):
        vi = np.random.choice(len(train_valid), min(batch, len(train_valid)), replace=False)
        ci = np.random.choice(len(train_corr), min(batch, len(train_corr)), replace=False)
        vz0 = torch.stack([train_valid[i][0] for i in vi]).to(DEVICE)
        vA = torch.stack([train_valid[i][1] for i in vi]).to(DEVICE)
        vZ = torch.stack([train_valid[i][2] for i in vi]).to(DEVICE)
        cz0 = torch.stack([train_corr[ci[i]][0] for i in range(len(ci))]).to(DEVICE)
        cA = torch.stack([train_corr[ci[i]][1] for i in range(len(ci))]).to(DEVICE)
        cZ = torch.stack([train_corr[ci[i]][2] for i in range(len(ci))]).to(DEVICE)
        opt.zero_grad()
        ev = model(vz0, vA, vZ)
        ec = model(cz0, cA, cZ)
        loss = F.relu(ev + 1.0).mean() + F.relu(1.0 - ec).mean() + \
               0.001 * (ev.pow(2).mean() + ec.pow(2).mean())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if (ep + 1) % 50 == 0:
            print(f"  [{label}] ep {ep+1:3d}: ev={ev.mean().item():+.2f} ec={ec.mean().item():+.2f}")

    model.eval()
    return model


def evaluate(model, test_valid, test_corr_by_type):
    """Per-violation-type AUROC on test set."""
    model.eval()
    results = {}
    with torch.no_grad():
        valid_energies = []
        for z0, A, Z in test_valid:
            e = model(z0.unsqueeze(0).to(DEVICE), A.unsqueeze(0).to(DEVICE),
                      Z.unsqueeze(0).to(DEVICE)).item()
            valid_energies.append(e)

        for vt in sorted(test_corr_by_type.keys()):
            corr_energies = []
            for z0, A, Z in test_corr_by_type[vt]:
                e = model(z0.unsqueeze(0).to(DEVICE), A.unsqueeze(0).to(DEVICE),
                          Z.unsqueeze(0).to(DEVICE)).item()
                corr_energies.append(e)
            labels = [0] * len(valid_energies) + [1] * len(corr_energies)
            energies = valid_energies + corr_energies
            results[vt] = {
                'auroc': roc_auc_score(labels, energies),
                'ev': np.mean(valid_energies),
                'ei': np.mean(corr_energies),
            }
    return results


# ─── Load data ────────────────────────────────────────────────────────────────

set_seed(42)
with open("data/compositional_grid.pkl", "rb") as f:
    data = pickle.load(f)

train_valid = [(torch.from_numpy(z0).float(), torch.from_numpy(A).float(),
                torch.from_numpy(Z).float())
               for z0, A, Z in data["train_valid"]]
train_corr_raw = [(torch.from_numpy(item[0][0]).float(), torch.from_numpy(item[0][1]).float(),
                   torch.from_numpy(item[0][2]).float(), item[1])
                  for item in data["train_corr"]]
test_valid = [(torch.from_numpy(z0).float(), torch.from_numpy(A).float(),
               torch.from_numpy(Z).float())
              for z0, A, Z in data["test_valid"]]
test_corr_raw = [(torch.from_numpy(item[0][0]).float(), torch.from_numpy(item[0][1]).float(),
                  torch.from_numpy(item[0][2]).float(), item[1])
                 for item in data["test_corr"]]

N, d, H = MAX_OBJECTS, ORACLE_SLOT_DIM, 25
print(f"Train: {len(train_valid)} valid, {len(train_corr_raw)} corrupt")
print(f"Test:  {len(test_valid)} valid, {len(test_corr_raw)} corrupt")

test_by_type = defaultdict(list)
for z0, A, Z, meta in test_corr_raw:
    test_by_type[meta["violation_type"]].append((z0, A, Z))

# ─── Model B: Random corruptions (noise-based) ───────────────────────────────

print("\nGenerating RANDOM corruptions for Model B...")
n_corr = len(train_corr_raw)
random_corr = []
for i in range(n_corr):
    # Pick a random valid trajectory and perturb its slots
    idx = np.random.randint(0, len(train_valid))
    z0, A, Z = train_valid[idx]
    # Add Gaussian noise to state slots (identity channels stay fixed)
    noise_scale = 0.3
    Z_noisy = Z.clone() + torch.randn_like(Z) * noise_scale
    # Keep type (channels 0-4) and identity (channels 15-18) unchanged
    Z_noisy[:, :, :5] = Z[:, :, :5]
    Z_noisy[:, :, 15:] = Z[:, :, 15:]
    random_corr.append((z0.clone(), A.clone(), Z_noisy))

# ─── Model C: Compositional corruption grid ──────────────────────────────────

train_corr_structured = [(z0, A, Z) for z0, A, Z, _ in train_corr_raw]

# ─── Train both models ───────────────────────────────────────────────────────

EPOCHS = 150

print(f"\n[Model B] Training with RANDOM corruptions ({EPOCHS} epochs)...")
model_b = train_model(train_valid, random_corr, EPOCHS, "B", d, 11, N)

print(f"\n[Model C] Training with COMPOSITIONAL corruptions ({EPOCHS} epochs)...")
model_c = train_model(train_valid, train_corr_structured, EPOCHS, "C", d, 11, N)

# ─── Evaluate ────────────────────────────────────────────────────────────────

print("\n" + "=" * 70)
print(f"{'Type':<14} {'B AUROC':>10} {'C AUROC':>10} {'Delta':>10} {'Verdict':>12}")
print("-" * 70)

rb = evaluate(model_b, test_valid, test_by_type)
rc = evaluate(model_c, test_valid, test_by_type)

deltas = []
for vt in sorted(test_by_type.keys()):
    ab = rb[vt]['auroc']
    ac = rc[vt]['auroc']
    delta = ac - ab
    deltas.append(delta)
    verdict = "AC3 wins" if delta > 0.10 else ("tie" if abs(delta) < 0.05 else "random wins")
    print(f"{vt:<14} {ab:10.3f} {ac:10.3f} {delta:+10.3f} {verdict:>12}")

# Summary
print("-" * 70)
avg_delta = np.mean(deltas)
print(f"{'AVERAGE':<14} {np.mean([rb[vt]['auroc'] for vt in rb]):10.3f} "
      f"{np.mean([rc[vt]['auroc'] for vt in rc]):10.3f} {avg_delta:+10.3f}")

# Energy margin comparison
print(f"\n{'Model':<10} {'E_valid':>10} {'E_invalid':>10} {'Margin':>10}")
print("-" * 45)
for name, r in [("B", rb), ("C", rc)]:
    ev_m = np.mean([v['ev'] for v in r.values()])
    ei_m = np.mean([v['ei'] for v in r.values()])
    print(f"{name:<10} {ev_m:10.2f} {ei_m:10.2f} {ei_m-ev_m:10.2f}")

print("\n" + "=" * 70)
n_wins = sum(1 for d in deltas if d > 0.10)
n_losses = sum(1 for d in deltas if d < -0.10)

if n_wins >= 3 and avg_delta > 0.10:
    print(f"AC3 CURRICULUM WORKS — Model C beats B on {n_wins}/6 types (avg Δ={avg_delta:+.3f})")
    print("The compositional corruption grid produces better cross-surface generalization.")
elif avg_delta > 0.05:
    print(f"WEAK SIGNAL — Model C slightly ahead (avg Δ={avg_delta:+.3f})")
else:
    print(f"NO ADVANTAGE — Curriculum doesn't beat random corruptions (avg Δ={avg_delta:+.3f})")
