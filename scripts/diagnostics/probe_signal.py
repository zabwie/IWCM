#!/usr/bin/env python3
"""Two-probe diagnostic: does Z contain oracle-detectable causal invalidity?

Probe A: flat MLP on flattened(Z) — tests raw separability
Probe B: structured temporal-slot validator — tests whether signal needs architecture

Both trained with oracle valid/invalid labels. Reports:
  - Overall AUROC
  - Per-violation-type AUROC
  - Held-out violation type generalization
  - Interpretation table
"""
import sys, pickle, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from pathlib import Path
from collections import defaultdict
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.utils.seed import set_seed

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
D_SLOT = 19   # ORACLE_SLOT_DIM
N_SLOTS = 8   # MAX_OBJECTS
H = 25        # horizon
D_ACTION = 11
BATCH = 64
EPOCHS = 80
EPOCHS_LONG = 200
SEEDS = [42, 123, 456]

# ═══════════════════════════════════════════════════════════════════════════════
# Data loading
# ═══════════════════════════════════════════════════════════════════════════════

with open("data/compositional_grid.pkl", "rb") as f:
    raw = pickle.load(f)

train_valid = [(torch.from_numpy(z0).float(), torch.from_numpy(A).float(),
                 torch.from_numpy(Z).float()) for z0, A, Z in raw["train_valid"]]
train_corr = [(torch.from_numpy(e[0][0]).float(), torch.from_numpy(e[0][1]).float(),
               torch.from_numpy(e[0][2]).float(), e[1]) for e in raw["train_corr"]]
test_valid = [(torch.from_numpy(z0).float(), torch.from_numpy(A).float(),
               torch.from_numpy(Z).float()) for z0, A, Z in raw["test_valid"]]
test_corr = [(torch.from_numpy(e[0][0]).float(), torch.from_numpy(e[0][1]).float(),
              torch.from_numpy(e[0][2]).float(), e[1]) for e in raw["test_corr"]]

# Build per-type test sets
test_by_type = defaultdict(list)
for z0, A, Z, meta in test_corr:
    test_by_type[meta["violation_type"]].append((z0, A, Z))
VTYPES = sorted(test_by_type.keys())

# Build per-type train sets (for held-out generalization)
train_by_type = defaultdict(list)
for z0, A, Z, meta in train_corr:
    train_by_type[meta["violation_type"]].append((z0, A, Z))

print(f"Train: {len(train_valid)} valid, {len(train_corr)} corrupt")
print(f"Test:  {len(test_valid)} valid, {len(test_corr)} corrupt")
print(f"Violation types: {VTYPES}")
for vt in VTYPES:
    print(f"  {vt:<12} train={len(train_by_type[vt]):>3}  test={len(test_by_type[vt]):>3}")


# ═══════════════════════════════════════════════════════════════════════════════
# Probe A: Flat MLP on flattened Z
# ═══════════════════════════════════════════════════════════════════════════════

class FlatMLPValidator(nn.Module):
    """Probe A: flatten Z → MLP → binary valid/invalid."""
    def __init__(self, input_dim, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden // 2), nn.ReLU(),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, Z):
        B = Z.shape[0]
        return self.net(Z.reshape(B, -1)).squeeze(-1)  # (B,)


# ═══════════════════════════════════════════════════════════════════════════════
# Probe B: Structured temporal-slot validator
# ═══════════════════════════════════════════════════════════════════════════════

class StructuredValidator(nn.Module):
    """Probe B: per-slot temporal GRU → cross-slot attention → binary head."""
    def __init__(self, d_slot=D_SLOT, hidden=128):
        super().__init__()
        self.hidden = hidden
        self.temporal = nn.GRU(d_slot, hidden, num_layers=2, batch_first=True,
                                bidirectional=True, dropout=0.1)
        self.cross_slot_attn = nn.MultiheadAttention(hidden * 2, num_heads=4,
                                                      batch_first=True, dropout=0.1)
        self.head = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, 1),
        )

    def forward(self, Z):
        B, H_, N, d = Z.shape
        # Per-slot temporal encoding
        Z_r = Z.permute(0, 2, 1, 3).reshape(B * N, H_, d)  # (B*N, H, d)
        _, h_n = self.temporal(Z_r)
        # h_n: (2*layers, B*N, hidden) -> concatenate last forward/backward
        h_fwd = h_n[-2].reshape(B, N, self.hidden)   # (B, N, hidden)
        h_bwd = h_n[-1].reshape(B, N, self.hidden)
        h_cat = torch.cat([h_fwd, h_bwd], dim=-1)    # (B, N, 2*hidden)
        # Cross-slot attention
        h_attn, _ = self.cross_slot_attn(h_cat, h_cat, h_cat)  # (B, N, 2*hidden)
        # Pool over slots and classify
        h_pool = h_attn.mean(dim=1)  # (B, 2*hidden)
        return self.head(h_pool).squeeze(-1)  # (B,)


# ═══════════════════════════════════════════════════════════════════════════════
# Training
# ═══════════════════════════════════════════════════════════════════════════════

def train_validator(model, train_valid_data, train_corr_data, epochs, seed, label=""):
    """Train binary classifier: valid=0, corrupt=1."""
    set_seed(seed)
    model = model.to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    n_valid = len(train_valid_data)
    n_corr = len(train_corr_data)

    for ep in range(epochs):
        vi = np.random.choice(n_valid, min(BATCH, n_valid), replace=False)
        ci = np.random.choice(n_corr, min(BATCH, n_corr), replace=False)

        Z_v = torch.stack([train_valid_data[i][2] for i in vi]).to(DEVICE)
        Z_c = torch.stack([train_corr_data[i][2] for i in ci]).to(DEVICE)
        Z_batch = torch.cat([Z_v, Z_c], dim=0)
        labels = torch.cat([torch.zeros(len(vi)), torch.ones(len(ci))]).to(DEVICE)

        opt.zero_grad()
        logits = model(Z_batch)
        loss = F.binary_cross_entropy_with_logits(logits, labels)
        loss.backward()
        opt.step()

        if (ep + 1) % 40 == 0:
            with torch.no_grad():
                acc = ((logits > 0) == labels.bool()).float().mean()
            print(f"  [{label}] ep{ep+1:3d}: loss={loss:.4f}  acc={acc:.3f}")

    return model.eval()


def evaluate_validator(model, test_valid_data, test_by_type_data):
    """Per-violation-type AUROC against valid data."""
    model.eval()
    results = {}
    with torch.no_grad():
        # Score all valid
        valid_scores = []
        for z0, A, Z in test_valid_data:
            score = model(Z.unsqueeze(0).to(DEVICE)).item()
            valid_scores.append(score)

        for vt in sorted(test_by_type_data.keys()):
            corr_scores = []
            for z0, A, Z in test_by_type_data[vt]:
                score = model(Z.unsqueeze(0).to(DEVICE)).item()
                corr_scores.append(score)
            labels = [0] * len(valid_scores) + [1] * len(corr_scores)
            scores = valid_scores + corr_scores
            results[vt] = roc_auc_score(labels, scores)

    # Overall
    all_corr = []
    for vt in sorted(test_by_type_data.keys()):
        all_corr.extend(test_by_type_data[vt])
    all_corr_scores = []
    for z0, A, Z in all_corr:
        all_corr_scores.append(model(Z.unsqueeze(0).to(DEVICE)).item())
    overall_labels = [0] * len(valid_scores) + [1] * len(all_corr_scores)
    overall_scores = valid_scores + all_corr_scores
    results["OVERALL"] = roc_auc_score(overall_labels, overall_scores)

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# Held-out violation type generalization
# ═══════════════════════════════════════════════════════════════════════════════

def train_and_eval_heldout(violation_type, model_class, model_kwargs, epochs, probe_label):
    """Train on all violation types EXCEPT held_out, test on held_out."""
    held_out_vt = violation_type

    # Build train sets excluding held-out
    train_valid_ho = list(train_valid)
    train_corr_ho = [(z0, A, Z) for z0, A, Z, meta in train_corr
                     if meta["violation_type"] != held_out_vt]

    # Test is still the full test set (but we only report held-out type)
    test_ho = test_by_type.get(held_out_vt, [])

    set_seed(42)
    model = model_class(**model_kwargs).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    n_valid = len(train_valid_ho)
    n_corr = len(train_corr_ho)

    for ep in range(epochs):
        vi = np.random.choice(n_valid, min(BATCH, n_valid), replace=False)
        ci = np.random.choice(n_corr, min(BATCH, n_corr), replace=False)

        Z_v = torch.stack([train_valid_ho[i][2] for i in vi]).to(DEVICE)
        Z_c = torch.stack([train_corr_ho[i][2] for i in ci]).to(DEVICE)
        Z_batch = torch.cat([Z_v, Z_c], dim=0)
        labels = torch.cat([torch.zeros(len(vi)), torch.ones(len(ci))]).to(DEVICE)

        opt.zero_grad()
        logits = model(Z_batch)
        loss = F.binary_cross_entropy_with_logits(logits, labels)
        loss.backward()
        opt.step()

    # Evaluate on held-out type only
    model.eval()
    with torch.no_grad():
        valid_scores = []
        for z0, A, Z in test_valid:
            valid_scores.append(model(Z.unsqueeze(0).to(DEVICE)).item())

        if len(test_ho) == 0:
            return held_out_vt, None

        corr_scores = []
        for z0, A, Z in test_ho:
            corr_scores.append(model(Z.unsqueeze(0).to(DEVICE)).item())

        labels = [0] * len(valid_scores) + [1] * len(corr_scores)
        scores = valid_scores + corr_scores
        auroc = roc_auc_score(labels, scores)

    return held_out_vt, auroc


class SlotPoolingValidator(nn.Module):
    def __init__(self, d_slot=D_SLOT, hidden=128):
        super().__init__()
        self.slot_encoder = nn.Sequential(
            nn.Linear(d_slot * 3, hidden),
            nn.ReLU(),
        )
        self.cross_slot = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, 1),
        )

    def forward(self, Z):
        B, H_, N, d = Z.shape
        Z_mean = Z.mean(dim=1)
        Z_max = Z.amax(dim=1)
        Z_var = F.relu((Z * Z).mean(dim=1) - Z_mean * Z_mean)
        Z_std = torch.sqrt(Z_var + 1e-5)
        Z_pooled = torch.cat([Z_mean, Z_max, Z_std], dim=-1)
        slot_feats = self.slot_encoder(Z_pooled)
        combined = torch.cat([slot_feats.mean(dim=1), slot_feats.amax(dim=1)], dim=-1)
        return self.cross_slot(combined).squeeze(-1)

print("\n" + "=" * 75)
print("PROBE A: Flat MLP on flattened Z")
print("=" * 75)

flat_input_dim = H * N_SLOTS * D_SLOT
flat_results_all = []

for seed in SEEDS:
    print(f"\n  Seed {seed}...")
    model_a = FlatMLPValidator(input_dim=flat_input_dim)
    train_corr_z = [(z0, A, Z) for z0, A, Z, _ in train_corr]
    model_a = train_validator(model_a, train_valid, train_corr_z,
                              EPOCHS, seed, f"FlatMLP-{seed}")
    results = evaluate_validator(model_a, test_valid, test_by_type)
    flat_results_all.append(results)
    for vt in VTYPES + ["OVERALL"]:
        print(f"    {vt:<12} {results[vt]:.4f}")

# Average across seeds
print(f"\n  --- Multi-seed Average ---")
for vt in VTYPES + ["OVERALL"]:
    vals = [r[vt] for r in flat_results_all]
    print(f"    {vt:<12} {np.mean(vals):.4f} ± {np.std(vals):.4f}")

overall_flat = np.mean([r["OVERALL"] for r in flat_results_all])

print("\n" + "=" * 75)
print("PROBE B: Structured temporal-slot validator")
print("=" * 75)

struct_results_all = []

for seed in SEEDS:
    print(f"\n  Seed {seed}...")
    model_b = StructuredValidator(d_slot=D_SLOT, hidden=128)
    model_b = train_validator(model_b, train_valid,
                              [(z0, A, Z) for z0, A, Z, _ in train_corr],
                              EPOCHS, seed, f"Struct-{seed}")
    results = evaluate_validator(model_b, test_valid, test_by_type)
    struct_results_all.append(results)
    for vt in VTYPES + ["OVERALL"]:
        print(f"    {vt:<12} {results[vt]:.4f}")

print(f"\n  --- Multi-seed Average ---")
for vt in VTYPES + ["OVERALL"]:
    vals = [r[vt] for r in struct_results_all]
    print(f"    {vt:<12} {np.mean(vals):.4f} ± {np.std(vals):.4f}")

overall_struct = np.mean([r["OVERALL"] for r in struct_results_all])

print("\n" + "=" * 75)
print("PROBE C: Slot-pooling structured validator")
print("=" * 75)

slotpool_results_all = []

for seed in SEEDS:
    print(f"\n  Seed {seed}...")
    model_c = SlotPoolingValidator(d_slot=D_SLOT, hidden=128)
    model_c = train_validator(model_c, train_valid,
                              [(z0, A, Z) for z0, A, Z, _ in train_corr],
                              EPOCHS, seed, f"SlotPool-{seed}")
    results = evaluate_validator(model_c, test_valid, test_by_type)
    slotpool_results_all.append(results)
    for vt in VTYPES + ["OVERALL"]:
        print(f"    {vt:<12} {results[vt]:.4f}")

print(f"\n  --- Multi-seed Average ---")
for vt in VTYPES + ["OVERALL"]:
    vals = [r[vt] for r in slotpool_results_all]
    print(f"    {vt:<12} {np.mean(vals):.4f} ± {np.std(vals):.4f}")

overall_pool = np.mean([r["OVERALL"] for r in slotpool_results_all])

# ═══════════════════════════════════════════════════════════════════════════════
# Held-out violation type generalization
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 75)
print("HELD-OUT VIOLATION TYPE GENERALIZATION")
print("=" * 75)

print(f"\n{'Violation Type':<14} {'Flat MLP':>10} {'Struct':>10} {'SlotPool':>10}")
print("-" * 52)

ho_flat = {}
ho_struct = {}
ho_pool = {}
for vt in VTYPES:
    _, a_ho = train_and_eval_heldout(vt, FlatMLPValidator,
                                      dict(input_dim=flat_input_dim),
                                      EPOCHS, "Flat")
    _, s_ho = train_and_eval_heldout(vt, StructuredValidator,
                                      dict(d_slot=D_SLOT, hidden=128),
                                      EPOCHS, "Struct")
    _, p_ho = train_and_eval_heldout(vt, SlotPoolingValidator,
                                      dict(d_slot=D_SLOT, hidden=128),
                                      EPOCHS, "SlotPool")
    ho_flat[vt] = a_ho
    ho_struct[vt] = s_ho
    ho_pool[vt] = p_ho
    a_str = f"{a_ho:.4f}" if a_ho is not None else "N/A"
    s_str = f"{s_ho:.4f}" if s_ho is not None else "N/A"
    p_str = f"{p_ho:.4f}" if p_ho is not None else "N/A"
    print(f"{vt:<14} {a_str:>10} {s_str:>10} {p_str:>10}")

ho_flat_mean = np.mean([v for v in ho_flat.values() if v is not None])
ho_struct_mean = np.mean([v for v in ho_struct.values() if v is not None])
ho_pool_mean = np.mean([v for v in ho_pool.values() if v is not None])

print("\n" + "=" * 75)
print("SUMMARY TABLE: Per-type AUROC (multi-seed mean)")
print("=" * 75)

print(f"\n{'Type':<14} {'Flat MLP':>10} {'Struct':>10} {'SlotPool':>10}")
print("-" * 52)
for vt in VTYPES:
    a = np.mean([r[vt] for r in flat_results_all])
    s = np.mean([r[vt] for r in struct_results_all])
    p = np.mean([r[vt] for r in slotpool_results_all])
    best = max(a, s, p)
    marker = " <<" if best > 0.85 else ""
    print(f"{vt:<14} {a:10.4f} {s:10.4f} {p:10.4f}{marker}")

a = overall_flat
s = overall_struct
p = overall_pool
best = max(a, s, p)
print(f"{'OVERALL':<14} {a:10.4f} {s:10.4f} {p:10.4f}{' <<' if best > 0.85 else ''}")
print(f"\n{'Held-Out':<14} {ho_flat_mean:10.4f} {ho_struct_mean:10.4f} {ho_pool_mean:10.4f}")

# ═══════════════════════════════════════════════════════════════════════════════
# Interpretation
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 75)
print("INTERPRETATION")
print("=" * 75)

best_overall = max(overall_flat, overall_struct, overall_pool)
best_heldout = max(ho_flat_mean, ho_struct_mean, ho_pool_mean)
best_probe = "Flat MLP" if best_overall == overall_flat else \
             ("Structured GRU" if best_overall == overall_struct else "SlotPool")

print(f"""
  Best overall AUROC:   {best_overall:.4f} ({best_probe})
  Best held-out AUROC:  {best_heldout:.4f}

  Per-type highlights (best architecture per type):
""")
for vt in VTYPES:
    vals = [
        ("Flat", np.mean([r[vt] for r in flat_results_all])),
        ("Struct", np.mean([r[vt] for r in struct_results_all])),
        ("SlotPool", np.mean([r[vt] for r in slotpool_results_all])),
    ]
    best_name, best_val = max(vals, key=lambda x: x[1])
    print(f"    {vt:<12} {best_val:.4f} ({best_name})")

if best_overall >= 0.90:
    verdict = "SIGNAL TRIVIAL — causal invalidity is nearly linearly separable."
    action = "TAMG failure is purely validator/corruptor extraction. Fix TAMG, not encoder."
elif best_overall >= 0.80:
    verdict = "SIGNAL PRESENT — Z encodes causal validity moderately well."
    action = "The signal exists. Structured validators + more data/epochs should close the gap. TAMG is viable."
elif best_overall >= 0.70:
    verdict = "SIGNAL WEAK — Z has some validity signal but it is noisy."
    action = "Consider encoder improvements AND better validator architecture. TAMG is high-risk."
else:
    verdict = "Z IS CAUSALLY WEAK — current latent representation doesn't encode causal validity."
    action = "TAMG cannot fix this. Go back to encoder. Improve slot representation or corruption curriculum."

print(f"\n  Verdict: {verdict}")
print(f"  Action:  {action}")
print()

print(f"\n  Gate 1 (supervised validator AUROC >= 0.90): {'PASSED' if best_overall >= 0.90 else 'FAILED'}")
gap = max(0, 0.90 - best_overall)
if gap > 0:
    print(f"    Gap to 0.90: {gap:.4f}")

print(f"\n  Held-out generalization (>= 0.70):           {'STRONG' if best_heldout >= 0.70 else 'WEAK'}")
if best_heldout < 0.70:
    print(f"    Gap to 0.70: {max(0, 0.70 - best_heldout):.4f}")
    print(f"    Model may be learning surface patterns, not causal invariants.")

print(f"\n  Recommendation:")
if best_overall >= 0.85 and best_heldout >= 0.65:
    print(f"    PROCEED — Z contains sufficient signal. Build proper committee.")
elif best_overall >= 0.75:
    print(f"    CAUTIOUS — Signal is present. Train longer, add more data, or try deeper architectures.")
else:
    print(f"    BLOCKED — Signal is too weak. Revisit encoder or data generation.")
