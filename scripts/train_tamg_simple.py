#!/usr/bin/env python3
"""Train SimpleTAMG — self-supervised corruption detection (no oracle).

Compares self-supervised SimpleTAMG against oracle-supervised SlotIWCMEnergy
on the compositional corruption grid test split.

Key claim: compositional mechanical corruptions + anchored contrastive loss
can match oracle-labeled training for causal law detection.

Usage:
    python scripts/train_tamg_simple.py [--epochs 500] [--lr 3e-3] [--seed 42]
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import numpy as np
import torch
from collections import defaultdict
from sklearn.metrics import roc_auc_score

from src.utils.seed import set_seed
from src.iwcm.slot_energy import SlotIWCMEnergy
from src.env.oracle_slot_encoder import ORACLE_SLOT_DIM, MAX_OBJECTS
from src.tamg_simple import SimpleTAMG, load_compositional_grid

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ═══════════════════════════════════════════════════════════════════════════════
# Oracle-supervised baseline
# ═══════════════════════════════════════════════════════════════════════════════

def train_oracle_baseline(train_valid, train_corr, epochs, lr, seed):
    """SlotIWCMEnergy trained with oracle-labeled corruptions."""
    set_seed(seed)
    model = SlotIWCMEnergy(d_slot=ORACLE_SLOT_DIM, d_action=11,
                           hidden_dim=128, num_slots=MAX_OBJECTS).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    batch = 32

    corr_struct = [(z0, A, Z) for z0, A, Z, _ in train_corr]
    nv, nc = len(train_valid), len(corr_struct)

    for ep in range(epochs):
        vi = np.random.choice(nv, min(batch, nv), replace=False)
        ci = np.random.choice(nc, min(batch, nc), replace=False)

        vz0 = torch.stack([train_valid[i][0] for i in vi]).to(DEVICE)
        vA  = torch.stack([train_valid[i][1] for i in vi]).to(DEVICE)
        vZ  = torch.stack([train_valid[i][2] for i in vi]).to(DEVICE)
        cz0 = torch.stack([corr_struct[i][0] for i in ci]).to(DEVICE)
        cA  = torch.stack([corr_struct[i][1] for i in ci]).to(DEVICE)
        cZ  = torch.stack([corr_struct[i][2] for i in ci]).to(DEVICE)

        opt.zero_grad()
        Ev = model(vz0, vA, vZ)
        Ec = model(cz0, cA, cZ)
        loss = (torch.relu(Ev + 1.0).mean() +
                torch.relu(1.0 - Ec).mean() +
                0.001 * (Ev.pow(2).mean() + Ec.pow(2).mean()))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

    model.eval()
    return model


# ═══════════════════════════════════════════════════════════════════════════════
# Self-supervised SimpleTAMG
# ═══════════════════════════════════════════════════════════════════════════════

def train_self_supervised(train_valid, epochs, lr, seed):
    """SimpleTAMG — valid worldlines + mechanical corruptions only. No oracle.

    Pre-generates K corruptions per valid worldline to create a fixed training
    set. This reduces epoch-to-epoch variance and lets the model converge more
    stably than on-the-fly corruption generation.
    """
    set_seed(seed)
    model = SimpleTAMG(d_slot=ORACLE_SLOT_DIM, d_action=11,
                       hidden=128, num_slots=MAX_OBJECTS).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    nv = len(train_valid)
    batch = 32

    all_v = [(z0.to(DEVICE), A.to(DEVICE), Z.to(DEVICE))
             for z0, A, Z in train_valid]

    for ep in range(epochs):
        n_batches = max(1, nv // batch)
        for _ in range(n_batches):
            vi = np.random.choice(nv, min(batch, nv), replace=False)
            vz0 = torch.stack([all_v[i][0] for i in vi])
            vA  = torch.stack([all_v[i][1] for i in vi])
            vZ  = torch.stack([all_v[i][2] for i in vi])

            opt.zero_grad()
            loss = model.training_step(vz0, vA, vZ)
            if loss.item() == 0.0:
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        if (ep + 1) % 50 == 0:
            with torch.no_grad():
                vi = np.random.choice(nv, min(batch, nv), replace=False)
                vz0 = torch.stack([all_v[i][0] for i in vi])
                vA  = torch.stack([all_v[i][1] for i in vi])
                vZ  = torch.stack([all_v[i][2] for i in vi])
                Ev = model.energy_fn(vz0, vA, vZ)
            print(f"  Epoch {ep+1:4d}: E_valid={Ev.mean().item():.3f}")

    model.eval()
    return model


# ═══════════════════════════════════════════════════════════════════════════════
# Evaluation
# ═══════════════════════════════════════════════════════════════════════════════

def evaluate_model(model, test_valid, test_corr):
    """AUROC and per-violation-type breakdown on test split."""
    valid_scores = []
    for z0, A, Z in test_valid:
        s = model(z0.unsqueeze(0).to(DEVICE), A.unsqueeze(0).to(DEVICE),
                  Z.unsqueeze(0).to(DEVICE)).item()
        valid_scores.append(s)

    by_type = defaultdict(list)
    for z0, A, Z, meta in test_corr:
        by_type[meta["violation_type"]].append((z0, A, Z))

    results = {}
    for vtype, items in sorted(by_type.items()):
        corr_scores = [model(z0.unsqueeze(0).to(DEVICE), A.unsqueeze(0).to(DEVICE),
                             Z.unsqueeze(0).to(DEVICE)).item()
                       for z0, A, Z in items]
        labels = [0] * len(valid_scores) + [1] * len(corr_scores)
        results[vtype] = roc_auc_score(labels, valid_scores + corr_scores)

    # Overall
    all_corr = [model(z0.unsqueeze(0).to(DEVICE), A.unsqueeze(0).to(DEVICE),
                      Z.unsqueeze(0).to(DEVICE)).item()
                for items in by_type.values() for z0, A, Z in items]
    labels_all = [0] * len(valid_scores) + [1] * len(all_corr)
    results["overall"] = roc_auc_score(labels_all, valid_scores + all_corr)

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--baseline", action="store_true",
                        help="Train oracle-supervised baseline for comparison")
    args = parser.parse_args()

    print("=" * 70)
    print("SimpleTAMG — Self-Supervised Corruption Detection")
    print(f"Device: {DEVICE}  Epochs: {args.epochs}  LR: {args.lr}")
    print("=" * 70)

    print("\nLoading compositional grid...")
    data = load_compositional_grid("data/compositional_grid.pkl")
    print(f"  train_valid: {len(data['train_valid'])}  "
          f"train_corr: {len(data['train_corr'])}")
    print(f"  test_valid:  {len(data['test_valid'])}  "
          f"test_corr:  {len(data['test_corr'])}")

    # ─── Self-Supervised ──────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("SimpleTAMG (self-supervised — mechanical corruptions only)")
    print(f"{'='*70}")
    model_ss = train_self_supervised(
        data["train_valid"], args.epochs, args.lr, args.seed)

    results_ss = evaluate_model(model_ss, data["test_valid"], data["test_corr"])
    print(f"\n  {'Violation':<14s} {'AUROC':>8s}")
    print(f"  {'-'*22}")
    for vtype, auc in sorted(results_ss.items()):
        marker = " ★" if vtype == "overall" else ""
        print(f"  {vtype:<14s} {auc:8.4f}{marker}")

    # ─── Oracle Baseline ──────────────────────────────────────────────────
    if args.baseline:
        print(f"\n{'='*70}")
        print("Oracle-supervised baseline (SlotIWCMEnergy)")
        print(f"{'='*70}")
        model_oracle = train_oracle_baseline(
            data["train_valid"], data["train_corr"],
            args.epochs, args.lr, args.seed)
        results_oracle = evaluate_model(
            model_oracle, data["test_valid"], data["test_corr"])

        # ─── Comparison ──────────────────────────────────────────────────
        print(f"\n{'='*70}")
        print("SELF-SUPERVISED vs ORACLE-SUPERVISED")
        print(f"{'='*70}")
        print(f"  {'Violation':<14s} {'Self-Sup':>10s} {'Oracle':>10s} {'Δ':>10s}")
        print(f"  {'-'*46}")
        for vtype in sorted(results_ss.keys()):
            ss = results_ss.get(vtype, 0)
            oracle = results_oracle.get(vtype, 0)
            delta = ss - oracle
            sym = "✓" if delta >= -0.03 else ("~" if delta >= -0.08 else "✗")
            print(f"  {vtype:<14s} {ss:10.4f} {oracle:10.4f} {delta:+10.4f} {sym}")

        ss_ov = results_ss["overall"]
        or_ov = results_oracle["overall"]
        gap = ss_ov - or_ov
        if gap >= -0.02:
            print(f"\n  VERDICT: Self-supervised matches oracle ({gap:+.4f}). ✓")
        elif gap >= -0.05:
            print(f"\n  VERDICT: Close ({gap:+.4f}). ~")
        else:
            print(f"\n  VERDICT: Gap remains ({gap:+.4f}). ✗")

    # ─── Final ────────────────────────────────────────────────────────────
    overall = results_ss.get("overall", 0)
    target = 0.95
    print(f"\n  TARGET: {target:.2f}  ACHIEVED: {overall:.4f}  "
          f"{'✓' if overall >= target else '✗'}")

    ckpt = "outputs/checkpoints/tamg_simple.pt"
    Path(ckpt).parent.mkdir(parents=True, exist_ok=True)
    torch.save(model_ss.state_dict(), ckpt)
    print(f"  Saved: {ckpt}")


if __name__ == "__main__":
    main()
