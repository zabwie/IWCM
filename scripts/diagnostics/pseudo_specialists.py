#!/usr/bin/env python3
"""Path A kill-switch: self-supervised pseudo-specialist committee.

Generates 8 perturbation families (no oracle labels), applies manifold filter,
trains SlotPool specialists on each family vs valid, evaluates on oracle-invalid
test suite. Key question: can pseudo-negatives close the 0.832→0.979 gap while
preserving oracle-like response geometry?
"""
import sys, pickle, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from pathlib import Path
from collections import defaultdict
from sklearn.metrics import roc_auc_score, balanced_accuracy_score
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
D_SLOT = 19; N = 8; H_ = 25; D_ACTION = 11
BATCH = 64; EPOCHS = 100; SEED = 42

def set_seed(s):
    np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(s)

# ═══════════════════════════════════════════════════════════════════════════════
# SlotPool (proven architecture)
# ═══════════════════════════════════════════════════════════════════════════════

class SlotPoolValidator(nn.Module):
    def __init__(self, d_slot=D_SLOT, hidden=128):
        super().__init__()
        self.slot_encoder = nn.Sequential(nn.Linear(d_slot * 3, hidden), nn.ReLU())
        self.cross_slot = nn.Sequential(
            nn.Linear(hidden * 2, hidden), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(hidden, 1),
        )
    def forward(self, Z):
        B, H, Ns, d = Z.shape
        m = Z.mean(dim=1); mx = Z.amax(dim=1)
        v = F.relu((Z * Z).mean(dim=1) - m * m)
        Zp = torch.cat([m, mx, torch.sqrt(v + 1e-5)], dim=-1)
        sf = self.slot_encoder(Zp)
        return self.cross_slot(torch.cat([sf.mean(dim=1), sf.amax(dim=1)], dim=-1)).squeeze(-1)

# ═══════════════════════════════════════════════════════════════════════════════
# Data
# ═══════════════════════════════════════════════════════════════════════════════

with open("data/compositional_grid.pkl", "rb") as f:
    raw = pickle.load(f)

train_valid = [(torch.from_numpy(z0).float(), torch.from_numpy(A).float(),
                 torch.from_numpy(Z).float()) for z0, A, Z in raw["train_valid"]]
test_valid  = [(torch.from_numpy(z0).float(), torch.from_numpy(A).float(),
                 torch.from_numpy(Z).float()) for z0, A, Z in raw["test_valid"]]
test_corr_raw = [(torch.from_numpy(e[0][0]).float(), torch.from_numpy(e[0][1]).float(),
                  torch.from_numpy(e[0][2]).float(), e[1]) for e in raw["test_corr"]]
test_by_type = defaultdict(list)
test_corr_flat = []
for z0, A, Z, meta in test_corr_raw:
    test_by_type[meta["violation_type"]].append((z0, A, Z))
    test_corr_flat.append((z0, A, Z))
VTYPES = sorted(test_by_type.keys())

# Valid data statistics for manifold filter
valid_Zs = torch.stack([t[2] for t in train_valid])  # (Nv, H, 8, 19)
valid_norms = valid_Zs.norm(dim=-1)    # (Nv, H, 8)
valid_mean_norm = valid_norms.mean().item()
valid_std_norm = valid_norms.std().item()
valid_active = (valid_norms > 0.1).float().sum(dim=-1).float()  # active slots per frame
valid_mean_active = valid_active.mean().item()
valid_std_active = valid_active.std().item()

# Build kNN reference for manifold distance
valid_Z_flat = valid_Zs.reshape(-1, H_ * N * D_SLOT)  # (Nv, 3800)

print(f"Train: {len(train_valid)} valid")
print(f"Test:  {len(test_valid)} valid, {len(test_corr_flat)} corrupt")
print(f"Valid norm: {valid_mean_norm:.3f} ± {valid_std_norm:.3f}")
print(f"Valid active slots: {valid_mean_active:.2f} ± {valid_std_active:.2f}")

# ═══════════════════════════════════════════════════════════════════════════════
# Perturbation families (self-supervised, NO oracle labels)
# ═══════════════════════════════════════════════════════════════════════════════

def perturb_splice(Z):
    t = np.random.randint(1, H_)
    Zc = Z.clone()
    Zc[t:] = Zc[:H_ - t].clone()
    return Zc, "splice"

def perturb_swap(Z):
    i, j = np.random.choice(N, 2, replace=False)
    Zc = Z.clone()
    Zc[:, i], Zc[:, j] = Zc[:, j].clone(), Zc[:, i].clone()
    return Zc, "swap"

def perturb_drop(Z):
    i = np.random.randint(0, N)
    Zc = Z.clone()
    Zc[:, i] *= 0.0
    return Zc, "drop"

def perturb_dup(Z):
    i, j = np.random.choice(N, 2, replace=False)
    Zc = Z.clone()
    Zc[:, j] = Zc[:, i].clone()
    return Zc, "dup"

def perturb_drift(Z):
    noise = torch.randn_like(Z) * 0.03
    Zc = Z.clone()
    Zc[..., :5] += noise[..., :5]
    Zc[..., 7:9] += noise[..., 7:9]
    return Zc, "drift"

def perturb_teleport(Z):
    Zc = Z.clone()
    slot = np.random.randint(0, N)
    t = np.random.randint(1, H_ - 1)
    jump = (torch.rand(2) * 2 - 1) * 0.5
    Zc[t:, slot, 5] += jump[0]
    Zc[t:, slot, 6] += jump[1]
    return Zc, "teleport"

def perturb_action_mismatch(Z, A):
    Zc = Z.clone()
    if H_ >= 2:
        t = np.random.randint(0, H_ - 1)
        action_strength = A[t].abs().mean().item()
        active_slots = [i for i in range(N) if Zc[t, i].norm() > 0.1]
        inactive_slots = [i for i in range(N) if Zc[t, i].norm() <= 0.1]
        if inactive_slots:
            target = np.random.choice(inactive_slots)
            Zc[t + 1:, target] += action_strength * 0.2
    return Zc, "action_mismatch"

def perturb_noise_hard(Z, valid_Z_flat_ref, k=5):
    noise = torch.randn_like(Z) * 0.05
    Zc = Z + noise
    dist = ((Zc.reshape(-1) - valid_Z_flat_ref).pow(2).sum(dim=-1)).sqrt()
    knn_dist = dist.kthvalue(min(k, len(dist))).values.item()
    if knn_dist > 3 * valid_std_norm:
        Zc = Z + noise * 0.3
    return Zc, "noise_hard"

# ═══════════════════════════════════════════════════════════════════════════════
# Manifold filter
# ═══════════════════════════════════════════════════════════════════════════════

def passes_manifold_filter(Z_orig, Z_corr, verbose=False):
    perturbation_norm = (Z_corr - Z_orig).pow(2).mean().sqrt().item()
    if perturbation_norm > 1.0:
        return False

    corr_norms = Z_corr.norm(dim=-1)
    active_corr = (corr_norms > 0.1).float().sum(dim=-1).float()
    mean_active = active_corr.mean().item()
    if mean_active < max(1.0, valid_mean_active - 3 * valid_std_active):
        return False
    if mean_active > valid_mean_active + 3 * valid_std_active:
        return False

    return True

# ═══════════════════════════════════════════════════════════════════════════════
# Generate pseudo-corruptions for each family
# ═══════════════════════════════════════════════════════════════════════════════

FAMILIES = ["splice", "swap", "drop", "dup", "drift", "teleport", "action_mismatch", "noise_hard"]

def generate_pseudo_corruptions(valid_data, family_name, n_target=None):
    if n_target is None:
        n_target = len(valid_data)
    generated = []
    for z0, A, Z in valid_data:
        if family_name == "splice":
            Zc, _ = perturb_splice(Z)
        elif family_name == "swap":
            Zc, _ = perturb_swap(Z)
        elif family_name == "drop":
            Zc, _ = perturb_drop(Z)
        elif family_name == "dup":
            Zc, _ = perturb_dup(Z)
        elif family_name == "drift":
            Zc, _ = perturb_drift(Z)
        elif family_name == "teleport":
            Zc, _ = perturb_teleport(Z)
        elif family_name == "action_mismatch":
            Zc, _ = perturb_action_mismatch(Z, A)
        elif family_name == "noise_hard":
            Zc, _ = perturb_noise_hard(Z, valid_Z_flat)

        if passes_manifold_filter(Z, Zc):
            generated.append((z0.clone(), A.clone(), Zc))

    return generated

print("\nGenerating pseudo-corruptions (manifold-filtered)...")
pseudo_corruptions = {}
for fam in FAMILIES:
    corr = generate_pseudo_corruptions(train_valid, fam)
    pseudo_corruptions[fam] = corr
    print(f"  {fam:<16} {len(corr):>4} passed filter  ({len(corr)/max(len(train_valid),1)*100:.0f}%)")

# Mixed negatives for generalist
all_pseudo = []
for fam in FAMILIES:
    all_pseudo.extend(pseudo_corruptions[fam])
print(f"  mixed           {len(all_pseudo):>4} total")

# ═══════════════════════════════════════════════════════════════════════════════
# Training
# ═══════════════════════════════════════════════════════════════════════════════

def train_binary_validator(pos_data, neg_data, epochs, lr=1e-3):
    set_seed(SEED)
    model = SlotPoolValidator().to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    n_pos, n_neg = len(pos_data), len(neg_data)
    for ep in range(epochs):
        pi = np.random.choice(n_pos, min(BATCH, n_pos), replace=False)
        ni = np.random.choice(n_neg, min(BATCH, n_neg), replace=False)
        Z_p = torch.stack([pos_data[i][2] for i in pi]).to(DEVICE)
        Z_n = torch.stack([neg_data[i][2] for i in ni]).to(DEVICE)
        Z_batch = torch.cat([Z_p, Z_n], dim=0)
        labels = torch.cat([torch.zeros(len(pi)), torch.ones(len(ni))]).to(DEVICE)
        opt.zero_grad()
        loss = F.binary_cross_entropy_with_logits(model(Z_batch), labels)
        loss.backward(); opt.step()
    return model.eval()

@torch.no_grad()
def score_all(model, dataset):
    scores = []
    for z0, A, Z in dataset:
        scores.append(torch.sigmoid(model(Z.unsqueeze(0).to(DEVICE))).item())
    return np.array(scores)

print("\n=== Training pseudo-specialists ===")
pseudo_specialists = {}
for fam in FAMILIES:
    neg = pseudo_corruptions[fam]
    if len(neg) < 10:
        print(f"  SKIP {fam}: only {len(neg)} negatives")
        continue
    print(f"  V_{fam:<14} ({len(neg)} negatives)")
    model = train_binary_validator(train_valid, neg, EPOCHS)
    pseudo_specialists[fam] = model

# Pseudo-generalist (mixed negatives)
print(f"\n  V_mixed ({len(all_pseudo)} negatives)")
pseudo_generalist = train_binary_validator(train_valid, all_pseudo, EPOCHS)

# V_contrast baseline (crude self-supervised from prior experiment)
contrast_negs = []
for z0, A, Z in train_valid:
    Z_c = Z.clone()
    strategy = np.random.randint(0, 5)
    if strategy == 0: Z_c += torch.randn_like(Z_c) * 0.1
    elif strategy == 1:
        t = np.random.randint(0, H_); Z_c[t:] = Z_c[t:].flip(dims=[0])
    elif strategy == 2 and N >= 2:
        i, j = np.random.choice(N, 2, replace=False)
        Z_c[:, i], Z_c[:, j] = Z_c[:, j].clone(), Z_c[:, i].clone()
    elif strategy == 3:
        Z_c[:, np.random.randint(0, N)] *= 0.1
    elif strategy == 4:
        t_split = np.random.randint(1, H_); Z_c[t_split:] = Z_c[:H_ - t_split].clone()
    if passes_manifold_filter(Z, Z_c):
        contrast_negs.append((z0.clone(), A.clone(), Z_c))
print(f"\n  V_contrast ({len(contrast_negs)} negatives)")
pseudo_contrast = train_binary_validator(train_valid, contrast_negs, EPOCHS)

# ═══════════════════════════════════════════════════════════════════════════════
# Evaluate on oracle-invalid test suite
# ═══════════════════════════════════════════════════════════════════════════════

# Score all validators
fam_list = [f for f in FAMILIES if f in pseudo_specialists] + ["mixed", "contrast"]
val_list = [pseudo_specialists.get(f) for f in FAMILIES if f in pseudo_specialists] + \
           [pseudo_generalist, pseudo_contrast]

all_val_scores = {}
for name in fam_list:
    model = pseudo_specialists.get(name) or \
            (pseudo_generalist if name == "mixed" else pseudo_contrast)
    all_val_scores[name] = {
        "valid": score_all(model, test_valid),
        "corr_all": score_all(model, test_corr_flat),
    }
    for vt in VTYPES:
        all_val_scores[name][vt] = score_all(model, test_by_type[vt])

# ═══════════════════════════════════════════════════════════════════════════════
# Results
# ═══════════════════════════════════════════════════════════════════════════════

# 1. Per-type AUROC matrix
print("\n" + "=" * 90)
print("PER-TYPE AUROC MATRIX (pseudo-specialists on oracle-invalid)")
print("=" * 90)

header = f"{'Type':<14}"
for name in fam_list:
    header += f" {name[:7]:>8}"
print(f"\n{header}")
print("-" * (14 + 9 * len(fam_list)))

for vt in VTYPES:
    row = f"{vt:<14}"
    row_scores = []
    for name in fam_list:
        scores_valid = all_val_scores[name]["valid"]
        scores_corr = all_val_scores[name][vt]
        labels = np.concatenate([np.zeros(len(scores_valid)), np.ones(len(scores_corr))])
        all_s = np.concatenate([scores_valid, scores_corr])
        a = roc_auc_score(labels, all_s)
        if a < 0.5: a = 1 - a
        row_scores.append(a)
        row += f" {a:8.4f}"
    print(row)

print(f"{'MEAN':<14}", end="")
for name in fam_list:
    vals = [roc_auc_score(
        np.concatenate([np.zeros(len(all_val_scores[name]["valid"])),
                         np.ones(len(all_val_scores[name][vt]))]),
        np.concatenate([all_val_scores[name]["valid"], all_val_scores[name][vt]])
    ) for vt in VTYPES]
    vals = [max(v, 1-v) for v in vals]
    print(f" {np.mean(vals):8.4f}", end="")
print()

# Overall binary AUROC per validator
print(f"\n{'Validator':<16} {'AUROC':>8}")
print("-" * 26)
for name in fam_list:
    sv = all_val_scores[name]["valid"]
    sc = all_val_scores[name]["corr_all"]
    labels = np.concatenate([np.zeros(len(sv)), np.ones(len(sc))])
    all_s = np.concatenate([sv, sc])
    a = roc_auc_score(labels, all_s)
    if a < 0.5: a = 1 - a
    print(f"  {name:<14} {a:8.4f}")

# Committee aggregation
fam_names_spec = [f for f in FAMILIES if f in pseudo_specialists]
n_spec = len(fam_names_spec)

spec_valid_mat = np.array([all_val_scores[n]["valid"] for n in fam_names_spec])
spec_corr_mat = np.array([all_val_scores[n]["corr_all"] for n in fam_names_spec])

# Add generalist
all_valid_mat = np.vstack([spec_valid_mat, all_val_scores["mixed"]["valid"][np.newaxis, :]])
all_corr_mat = np.vstack([spec_corr_mat, all_val_scores["mixed"]["corr_all"][np.newaxis, :]])

y_valid = np.zeros(len(test_valid))
y_corr = np.ones(len(test_corr_flat))
y_all = np.concatenate([y_valid, y_corr])

for label, v_mat, c_mat in [("Specialists only", spec_valid_mat, spec_corr_mat),
                              ("+ Generalist", all_valid_mat, all_corr_mat)]:
    mv = v_mat.mean(axis=0); mc = c_mat.mean(axis=0)
    xv = v_mat.max(axis=0); xc = c_mat.max(axis=0)
    vv = v_mat.var(axis=0); vc = c_mat.var(axis=0)

    print(f"\n  {label} committee AUROC:")
    for sig_name, sig in [
        ("Mean  ", np.concatenate([mv, mc])),
        ("Max   ", np.concatenate([xv, xc])),
        ("Var   ", np.concatenate([vv, vc])),
    ]:
        print(f"    {sig_name}: {roc_auc_score(y_all, sig):.4f}")

# ═══════════════════════════════════════════════════════════════════════════════
# Artifact detection check
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 90)
print("ARTIFACT DETECTION CHECK")
print("=" * 90)

pert_norms = []
for z0, A, Z in test_corr_flat:
    pert_norms.append(0.0)

for name in fam_list:
    scores = all_val_scores[name]["corr_all"]
    print(f"  {name:<14} mean_score={scores.mean():.4f}")

# Check if pseudo-specialist scores correlate with perturbation norm on VALID data
# (HIGH score on valid = false positive, should NOT be predictable from norm)
valid_norms_test = [Z.norm().item() for _, _, Z in test_valid]
for name in fam_list[:3]:
    sv = all_val_scores[name]["valid"]
    if len(sv) > 1:
        corr_valid_norm = np.corrcoef(sv, valid_norms_test)[0, 1]
        print(f"  {name:<14} corr(score, valid_norm)={corr_valid_norm:.4f}")

# Compare against oracle-specialist reference values
print(f"\n  Reference (oracle-specialist):")
print(f"    V_contrast baseline:  0.832")
print(f"    Oracle specialist max: 0.979")
print(f"    Target:               >= 0.90 (pseudo-specialist max)")

# ═══════════════════════════════════════════════════════════════════════════════
# Vote on Path A viability
# ═══════════════════════════════════════════════════════════════════════════════

best_spec = 0
best_name = ""
for name in fam_names_spec:
    sv = all_val_scores[name]["valid"]
    sc = all_val_scores[name]["corr_all"]
    labels = np.concatenate([np.zeros(len(sv)), np.ones(len(sc))])
    all_s = np.concatenate([sv, sc])
    a = roc_auc_score(labels, all_s)
    if a < 0.5: a = 1 - a
    if a > best_spec:
        best_spec = a
        best_name = name

pseudo_gen_auroc = roc_auc_score(
    np.concatenate([np.zeros(len(all_val_scores["mixed"]["valid"])),
                     np.ones(len(all_val_scores["mixed"]["corr_all"]))]),
    np.concatenate([all_val_scores["mixed"]["valid"], all_val_scores["mixed"]["corr_all"]]))

pseudo_cont_auroc = roc_auc_score(
    np.concatenate([np.zeros(len(all_val_scores["contrast"]["valid"])),
                     np.ones(len(all_val_scores["contrast"]["corr_all"]))]),
    np.concatenate([all_val_scores["contrast"]["valid"], all_val_scores["contrast"]["corr_all"]]))

spec_max = spec_valid_mat.max(axis=0)
spec_max_c = spec_corr_mat.max(axis=0)
pseudo_committee_max = roc_auc_score(y_all, np.concatenate([spec_max, spec_max_c]))

print("\n" + "=" * 90)
print("PATH A VIABILITY ASSESSMENT")
print("=" * 90)

print(f"""
  V_contrast baseline:          {pseudo_cont_auroc:.4f}
  Pseudo generalist:            {pseudo_gen_auroc:.4f}
  Best pseudo specialist:       {best_spec:.4f} ({best_name})
  Pseudo committee max:         {pseudo_committee_max:.4f}
  Oracle specialist max:        0.9790
  Gap (pseudo max → oracle):    {0.979 - pseudo_committee_max:.4f}
""")

if pseudo_committee_max >= 0.90:
    print("  VERDICT: Path A passes kill-switch. Pseudo-specialists close most of the gap.")
    print("  Paper claim: self-supervised pseudo-negatives approximate oracle causal geometry.")
elif pseudo_committee_max >= 0.85:
    print("  VERDICT: Marginal. Pseudo-specialists improve over contrast baseline.")
    print("  Paper claim: partial transfer; structural negatives needed for 0.90+.")
else:
    print("  VERDICT: Path A fails. Pseudo-negatives don't transfer to oracle invalidity.")
    print("  Honest conclusion: TAMG requires oracle-like structural negatives.")
