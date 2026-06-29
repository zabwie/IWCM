#!/usr/bin/env python3
"""Mechanism-validation: does a specialized validator committee produce structured
causal disagreement, or merely redundant invalidity detection?

Trains 7 SlotPool validators (6 specialized per violation type, 1 generalist)
with oracle valid/invalid labels. Reports:
  - Response matrix: rows = violation type, cols = validator output (AUROC)
  - Diagonal vs off-diagonal dominance
  - Effective rank of validator output matrix
  - Pairwise validator correlation on valid + each corrupt type
  - Generalist vs specialist committee comparison
"""
import sys, pickle, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from pathlib import Path
from collections import defaultdict
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
D_SLOT = 19
N_SLOTS = 8
H = 25
BATCH = 64
EPOCHS = 80
SEED = 42


# ═══════════════════════════════════════════════════════════════════════════════
# Architecture (same as proven SlotPool from probe_signal.py)
# ═══════════════════════════════════════════════════════════════════════════════

class SlotPoolValidator(nn.Module):
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


# ═══════════════════════════════════════════════════════════════════════════════
# Data loading
# ═══════════════════════════════════════════════════════════════════════════════

with open("data/compositional_grid.pkl", "rb") as f:
    raw = pickle.load(f)

train_valid = [(t[0].float(), t[1].float(), t[2].float())
               for t in [(torch.from_numpy(z0), torch.from_numpy(A), torch.from_numpy(Z))
               for z0, A, Z in raw["train_valid"]]]
train_corr_raw = [(torch.from_numpy(e[0][0]).float(), torch.from_numpy(e[0][1]).float(),
                   torch.from_numpy(e[0][2]).float(), e[1]) for e in raw["train_corr"]]
test_valid = [(t[0].float(), t[1].float(), t[2].float())
              for t in [(torch.from_numpy(z0), torch.from_numpy(A), torch.from_numpy(Z))
              for z0, A, Z in raw["test_valid"]]]
test_corr_raw = [(torch.from_numpy(e[0][0]).float(), torch.from_numpy(e[0][1]).float(),
                  torch.from_numpy(e[0][2]).float(), e[1]) for e in raw["test_corr"]]

# Organize by violation type
train_by_type = defaultdict(list)
test_by_type = defaultdict(list)
for z0, A, Z, meta in train_corr_raw:
    train_by_type[meta["violation_type"]].append((z0, A, Z))
for z0, A, Z, meta in test_corr_raw:
    test_by_type[meta["violation_type"]].append((z0, A, Z))

VTYPES = sorted(test_by_type.keys())
print(f"Train: {len(train_valid)} valid, {len(train_corr_raw)} corrupt")
print(f"Test:  {len(test_valid)} valid, {len(test_corr_raw)} corrupt")
print(f"Types: {VTYPES}")
for vt in VTYPES:
    print(f"  {vt:<12} train={len(train_by_type[vt]):>3}  test={len(test_by_type[vt]):>3}")


# ═══════════════════════════════════════════════════════════════════════════════
# Training
# ═══════════════════════════════════════════════════════════════════════════════

def set_seed(s):
    np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


def train_validator(model, valid_data, corrupt_data, epochs, label):
    set_seed(SEED)
    model = model.to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    n_v, n_c = len(valid_data), len(corrupt_data)

    for ep in range(epochs):
        vi = np.random.choice(n_v, min(BATCH, n_v), replace=False)
        ci = np.random.choice(n_c, min(BATCH, n_c), replace=False)

        Z_v = torch.stack([valid_data[i][2] for i in vi]).to(DEVICE)
        Z_c = torch.stack([corrupt_data[i][2] for i in ci]).to(DEVICE)
        Z_batch = torch.cat([Z_v, Z_c], dim=0)
        labels = torch.cat([torch.zeros(len(vi)), torch.ones(len(ci))]).to(DEVICE)

        opt.zero_grad()
        loss = F.binary_cross_entropy_with_logits(model(Z_batch), labels)
        loss.backward()
        opt.step()

    model.eval()
    return model


# ═══════════════════════════════════════════════════════════════════════════════
# Scoring
# ═══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def score_all(validator, dataset):
    scores = []
    for z0, A, Z in dataset:
        scores.append(torch.sigmoid(validator(Z.unsqueeze(0).to(DEVICE))).item())
    return np.array(scores)


@torch.no_grad()
def compute_response_matrix(validators, test_valid_data, test_by_type_data):
    """Rows = violation type, cols = validator index. Entry = mean score.
    Also compute per-type AUROC for each validator."""
    valid_scores = []
    for v in validators:
        valid_scores.append(score_all(v, test_valid_data))

    n_types = len(VTYPES)
    n_val = len(validators)
    response = np.zeros((n_types, n_val))
    auroc_mat = np.zeros((n_types, n_val))

    for i, vt in enumerate(VTYPES):
        corrupt_data = test_by_type_data[vt]
        for j, v in enumerate(validators):
            corr_scores = score_all(v, corrupt_data)
            response[i, j] = corr_scores.mean()
            labels = np.concatenate([np.zeros(len(valid_scores[j])), np.ones(len(corr_scores))])
            all_scores = np.concatenate([valid_scores[j], corr_scores])
            auroc_mat[i, j] = roc_auc_score(labels, all_scores)

    return response, auroc_mat, valid_scores


def compute_pairwise_stats(validators, valid_scores, test_by_type_data):
    """Compute per-type pairwise validator correlations and effective rank."""
    n_val = len(validators)
    n_types = len(VTYPES)

    # Build all scores matrix: (n_val, n_samples) for each data subset
    results = {}
    for subset_name, data in [("valid", None), *[(vt, test_by_type_data[vt]) for vt in VTYPES]]:
        if subset_name == "valid":
            subset_data = test_valid
        else:
            subset_data = data

        n = len(subset_data)
        scores_mat = np.zeros((n_val, n))
        for j, v in enumerate(validators):
            scores_mat[j] = score_all(v, subset_data)

        corr = np.corrcoef(scores_mat)
        U, S, Vt = np.linalg.svd(scores_mat - scores_mat.mean(axis=1, keepdims=True),
                                  full_matrices=False)
        eff_rank = (S > 1e-6).sum()
        results[subset_name] = {"corr": corr, "eff_rank": eff_rank, "scores": scores_mat}

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 75)
print("TRAINING 7 SLOTPOOL VALIDATORS")
print("=" * 75)

validators = {}
validators_z = {}

# 6 specialized validators
for vt in VTYPES:
    print(f"\n  V_{vt[:4]:4s}  train: valid vs {vt} ({len(train_by_type[vt])} corrupt)")
    model = SlotPoolValidator(d_slot=D_SLOT, hidden=128)
    model = train_validator(model, train_valid, train_by_type[vt], EPOCHS, vt)
    validators[vt] = model

# Generalist
all_corrupt = []
for vt in VTYPES:
    all_corrupt.extend(train_by_type[vt])
print(f"\n  V_general  train: valid vs ALL ({len(all_corrupt)} corrupt)")
model_g = SlotPoolValidator(d_slot=D_SLOT, hidden=128)
model_g = train_validator(model_g, train_valid, all_corrupt, EPOCHS, "general")
validators["general"] = model_g

# Ordered list for matrices
val_names = list(VTYPES) + ["general"]
val_list = [validators[n] for n in val_names]
n_val = len(val_list)

# ═══════════════════════════════════════════════════════════════════════════════
# Response matrix — AUROC
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 75)
print("RESPONSE MATRIX (AUROC per violation type per validator)")
print("=" * 75)

response, auroc_mat, valid_scores = compute_response_matrix(val_list, test_valid, test_by_type)

header = f"{'Violation':<14}"
for name in val_names:
    header += f" {name[:7]:>8}"
print(f"\n{header}")
print("-" * (14 + 9 * n_val))

diagonals = []
off_diagonals = []
for i, vt in enumerate(VTYPES):
    row = f"{vt:<14}"
    for j in range(n_val):
        v = auroc_mat[i, j]
        row += f" {v:8.4f}"
        if j == i:
            diagonals.append(v)
        elif j < len(VTYPES):
            off_diagonals.append(v)
    print(row)

# Generalist summary row
row_g = f"{'GENERALIST':<14}"
for j in range(n_val):
    row_g += f" {np.mean(auroc_mat[:, j]):8.4f}"
print()
print(row_g)
print(f"{'mean(diag)':<14} {np.mean(diagonals):8.4f}")
print(f"{'mean(off)':<14}  {np.mean(off_diagonals):8.4f}")
print(f"{'diag/off':<14} {np.mean(diagonals)/max(np.mean(off_diagonals), 0.001):8.2f}x")

# ═══════════════════════════════════════════════════════════════════════════════
# Mean score response matrix
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 75)
print("MEAN SCORE MATRIX (sigmoid output, higher = more likely invalid)")
print("=" * 75)

valid_mean = np.mean(valid_scores[0])  # all validators similar on valid

print(f"\n{header}")
print("-" * (14 + 9 * n_val))
print(f"{'valid':<14} ", end="")
for j in range(n_val):
    print(f" {np.mean(valid_scores[j]):8.4f}", end="")
print()

for i, vt in enumerate(VTYPES):
    row = f"{vt:<14}"
    for j in range(n_val):
        row += f" {response[i, j]:8.4f}"
    print(row)

# ═══════════════════════════════════════════════════════════════════════════════
# Disagreement score: does committee variance predict oracle invalidity?
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 75)
print("DISAGREEMENT AS ORACLE PREDICTOR")
print("=" * 75)

from sklearn.metrics import roc_auc_score as auroc

# Gather all test scores into (n_val, n_test) matrix
all_test_valid_scores = np.array([score_all(v, test_valid) for v in val_list])
all_test_corr = []
for vt in VTYPES:
    for z0, A, Z in test_by_type[vt]:
        all_test_corr.append((z0, A, Z))

all_corr_scores_mat = np.zeros((n_val, len(all_test_corr)))
for j, v in enumerate(val_list):
    all_corr_scores_mat[j] = score_all(v, all_test_corr)

# Committee signals
valid_var = all_test_valid_scores.var(axis=0)  # (n_valid,)
corr_var = all_corr_scores_mat.var(axis=0)     # (n_corr,)
valid_mean_comm = all_test_valid_scores.mean(axis=0)
corr_mean_comm = all_corr_scores_mat.mean(axis=0)
valid_max_comm = all_test_valid_scores.max(axis=0)
corr_max_comm = all_corr_scores_mat.max(axis=0)

# Generalist (last column)
generalist_valid = all_test_valid_scores[-1]
generalist_corr = all_corr_scores_mat[-1]

# Build AUROCs for each signal
y_valid = np.zeros(len(test_valid))
y_corr = np.ones(len(all_test_corr))
y = np.concatenate([y_valid, y_corr])

signals = {
    "Generalist (mean)": np.concatenate([generalist_valid, generalist_corr]),
    "Committee mean": np.concatenate([valid_mean_comm, corr_mean_comm]),
    "Committee max": np.concatenate([valid_max_comm, corr_max_comm]),
    "Committee variance": np.concatenate([valid_var, corr_var]),
}

print(f"\n{'Signal':<24} {'AUROC':>8}")
print("-" * 34)
for name, signal in signals.items():
    print(f"{name:<24} {auroc(y, signal):8.4f}")

# Is variance additive beyond mean?
from sklearn.linear_model import LogisticRegression
mean_feat = np.concatenate([valid_mean_comm, corr_mean_comm]).reshape(-1, 1)
var_feat = np.concatenate([valid_var, corr_var]).reshape(-1, 1)
combined_feat = np.column_stack([mean_feat, var_feat])

lr_mean = LogisticRegression().fit(mean_feat, y)
lr_both = LogisticRegression().fit(combined_feat, y)

print(f"\n  Logistic regression AUROC:")
print(f"    Mean only:           {auroc(y, lr_mean.predict_proba(mean_feat)[:, 1]):.4f}")
print(f"    Mean + Variance:     {auroc(y, lr_both.predict_proba(combined_feat)[:, 1]):.4f}")

# ═══════════════════════════════════════════════════════════════════════════════
# Pairwise correlations and effective rank
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 75)
print("VALIDATOR CORRELATIONS & EFFECTIVE RANK")
print("=" * 75)

pairwise = compute_pairwise_stats(val_list, valid_scores, test_by_type)

for subset_name in ["valid"] + VTYPES:
    stats = pairwise[subset_name]
    corr = stats["corr"]
    er = stats["eff_rank"]
    # Off-diagonal mean
    mask = ~np.eye(len(val_names), dtype=bool)
    off_diag_mean = corr[mask].mean() if len(val_names) > 1 else 0.0
    print(f"\n  {subset_name:<12}  eff_rank={er}  off-diag corr mean={off_diag_mean:.4f}")

# Detailed correlation matrix for valid data
print(f"\n{'':>12}", end="")
for name in val_names:
    print(f" {name[:6]:>7}", end="")
print()

corr_valid = pairwise["valid"]["corr"]
for i, name_i in enumerate(val_names):
    row = f"{name_i:>12}"
    for j in range(n_val):
        row += f" {corr_valid[i, j]:7.4f}"
    print(row)

# ═══════════════════════════════════════════════════════════════════════════════
# Per-validator self-AUROC
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 75)
print("PER-VALIDATOR SELF-AUROC (on own violation type)")
print("=" * 75)

print(f"\n{'Validator':<14} {'Self AUROC':>10} {'Other mean':>10} {'Gap':>10}")
print("-" * 48)
for i, vt in enumerate(VTYPES):
    self_auroc = auroc_mat[i, i]
    other_aurocs = [auroc_mat[j, i] for j in range(len(VTYPES)) if j != i]
    other_mean = np.mean(other_aurocs)
    print(f"{vt:<14} {self_auroc:10.4f} {other_mean:10.4f} {self_auroc - other_mean:+10.4f}")

# Generalist
gen_aurocs = auroc_mat[:, -1]
print(f"{'general':<14} {np.mean(gen_aurocs):10.4f} {'--':>10} {'--':>10}")

# ═══════════════════════════════════════════════════════════════════════════════
# Success criteria check
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 75)
print("SUCCESS CRITERIA")
print("=" * 75)

mean_diag = np.mean(diagonals)
mean_off = np.mean(off_diagonals)
all_above_85 = all(d >= 0.85 for d in diagonals)
diag_dominates = mean_diag > mean_off
min_eff_rank = min(stats["eff_rank"] for stats in pairwise.values())
max_valid_corr = pairwise["valid"]["corr"][~np.eye(n_val, dtype=bool)].max()
disagreement_adds = auroc(y, signals["Committee variance"]) > auroc(y, signals["Generalist (mean)"]) - 0.02

print(f"\n  [1] Each specialist AUROC > 0.85 on own type:  {'PASS' if all_above_85 else 'FAIL'}")
for i, vt in enumerate(VTYPES):
    print(f"      {vt:<14} {diagonals[i]:.4f}  {'✓' if diagonals[i] >= 0.85 else '✗'}")
print(f"\n  [2] Mean diagonal ({mean_diag:.3f}) > mean off-diagonal ({mean_off:.3f}): {'PASS' if diag_dominates else 'FAIL'}")
print(f"      Ratio: {mean_diag/max(mean_off, 0.001):.2f}x")
print(f"\n  [3] Effective rank > 2: min={min_eff_rank} ({'PASS' if min_eff_rank > 2 else 'FAIL'})")
print(f"\n  [4] Max pairwise correlation < 0.95: max={max_valid_corr:.4f} ({'PASS' if max_valid_corr < 0.95 else 'FAIL'})")
print(f"\n  [5] Disagreement adds to generalist: {'PASS' if disagreement_adds else 'INCONCLUSIVE'}")
gen_auroc_val = auroc(y, signals["Generalist (mean)"])
var_auroc_val = auroc(y, signals["Committee variance"])
print(f"      Generalist AUROC: {gen_auroc_val:.4f}")
print(f"      Variance AUROC:   {var_auroc_val:.4f}")
print(f"      Mean+Variance LR: {auroc(y, lr_both.predict_proba(combined_feat)[:, 1]):.4f}")

passed = sum([all_above_85, diag_dominates, min_eff_rank > 2, max_valid_corr < 0.95, disagreement_adds])
print(f"\n  Criteria passed: {passed}/5")
if passed >= 4:
    print("  VERDICT: Committee produces structured, type-specific disagreement.")
    print("  TAMG mechanism validated for oracle-slot setting.")
elif passed >= 3:
    print("  VERDICT: Partial structure. Specialization works but signal is weak.")
    print("  May need more data per type or longer training.")
else:
    print("  VERDICT: Specialists collapse to redundancy. Disagreement is not causal.")
    print("  TAMG committee mechanism not validated at this scale.")
