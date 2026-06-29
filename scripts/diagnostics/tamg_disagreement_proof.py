#!/usr/bin/env python3
"""TAMG disagreement mechanism proof: three decisive experiments.

1. TYPE CLASSIFICATION: Can the oracle-specialist committee response vector
   predict violation type? (Tests whether committee patterns encode causal semantics.)

2. RESIDUAL PATTERN: Does disagreement carry information after controlling
   for max confidence? (Tests whether variance > max for invalidity detection.)

3. CONDITIONAL FACTORIZATION: Do validators trained on different conditional
   objectives (valid-data-only, NO violation labels) produce structured rejection
   patterns on oracle-invalid data? (Tests the real TAMG hypothesis.)
"""
import sys, pickle, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from pathlib import Path
from collections import defaultdict
from sklearn.metrics import roc_auc_score, accuracy_score, balanced_accuracy_score
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
D_SLOT = 19; N_SLOTS = 8; H = 25; D_ACTION = 11
BATCH = 64; EPOCHS = 80; SEED = 42

def set_seed(s):
    np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(s)


# ═══════════════════════════════════════════════════════════════════════════════
# Shared architecture (proven SlotPool)
# ═══════════════════════════════════════════════════════════════════════════════

class SlotPoolValidator(nn.Module):
    def __init__(self, d_slot=D_SLOT, hidden=128, out_dim=1):
        super().__init__()
        self.slot_encoder = nn.Sequential(nn.Linear(d_slot * 3, hidden), nn.ReLU())
        self.cross_slot = nn.Sequential(
            nn.Linear(hidden * 2, hidden), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, Z):
        B, H_, N, d = Z.shape
        Z_mean = Z.mean(dim=1); Z_max = Z.amax(dim=1)
        Z_var = F.relu((Z * Z).mean(dim=1) - Z_mean * Z_mean)
        Z_std = torch.sqrt(Z_var + 1e-5)
        Z_pooled = torch.cat([Z_mean, Z_max, Z_std], dim=-1)
        slot_feats = self.slot_encoder(Z_pooled)
        combined = torch.cat([slot_feats.mean(dim=1), slot_feats.amax(dim=1)], dim=-1)
        return self.cross_slot(combined).squeeze(-1)


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

train_by_type = defaultdict(list)
test_by_type = defaultdict(list)
for z0, A, Z, meta in train_corr_raw:
    train_by_type[meta["violation_type"]].append((z0, A, Z))
for z0, A, Z, meta in test_corr_raw:
    test_by_type[meta["violation_type"]].append((z0, A, Z))

VTYPES = sorted(test_by_type.keys())
print(f"Train: {len(train_valid)}v + {len(train_corr_raw)}c, Test: {len(test_valid)}v + {len(test_corr_raw)}c")
print(f"Types: {VTYPES}")


# ═══════════════════════════════════════════════════════════════════════════════
# Train oracle-specialized committee (reuse from prior experiment)
# ═══════════════════════════════════════════════════════════════════════════════

def train_binary(model, pos_data, neg_data, epochs, lr=1e-3):
    set_seed(SEED)
    model = model.to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    n_pos, n_neg = len(pos_data), len(neg_data)
    for ep in range(epochs):
        pi = np.random.choice(n_pos, min(BATCH, n_pos), replace=False)
        ni = np.random.choice(n_neg, min(BATCH, n_neg), replace=False)
        Z_p = torch.stack([pos_data[i][2] for i in pi]).to(DEVICE)
        Z_n = torch.stack([neg_data[i][2] for i in ni]).to(DEVICE)
        Z_b = torch.cat([Z_p, Z_n], dim=0)
        y = torch.cat([torch.zeros(len(pi)), torch.ones(len(ni))]).to(DEVICE)
        opt.zero_grad(); loss = F.binary_cross_entropy_with_logits(model(Z_b), y)
        loss.backward(); opt.step()
    return model.eval()

@torch.no_grad()
def score_all(model, dataset):
    scores = []
    for z0, A, Z in dataset:
        scores.append(torch.sigmoid(model(Z.unsqueeze(0).to(DEVICE))).item())
    return np.array(scores)

print("\n=== Training oracle-specialized committee ===")
specialists = {}
for vt in VTYPES:
    model = SlotPoolValidator()
    model = train_binary(model, train_valid, train_by_type[vt], EPOCHS)
    specialists[vt] = model

all_corrupt = []
for vt in VTYPES: all_corrupt.extend(train_by_type[vt])
generalist = SlotPoolValidator()
generalist = train_binary(generalist, train_valid, all_corrupt, EPOCHS)

val_names = VTYPES + ["general"]
val_list = [specialists[vt] for vt in VTYPES] + [generalist]

# Build test score matrices
n_test_corr = len(test_corr_raw)
test_corr_flat = [(z0, A, Z) for z0, A, Z, _ in test_corr_raw]
test_corr_types = [meta["violation_type"] for _, _, _, meta in test_corr_raw]

spec_scores_valid = np.array([score_all(v, test_valid) for v in specialists.values()])
gen_scores_valid = score_all(generalist, test_valid)
spec_scores_corr = np.array([score_all(v, test_corr_flat) for v in specialists.values()])
gen_scores_corr = score_all(generalist, test_corr_flat)

# Full committee matrix
all_scores_valid = np.vstack([spec_scores_valid, gen_scores_valid[np.newaxis, :]])
all_scores_corr = np.vstack([spec_scores_corr, gen_scores_corr[np.newaxis, :]])

y_valid = np.zeros(len(test_valid))
y_corr = np.ones(n_test_corr)
y_all = np.concatenate([y_valid, y_corr])

# ═══════════════════════════════════════════════════════════════════════════════
# EXPERIMENT 1: Type classification from committee response vectors
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 75)
print("EXPERIMENT 1: TYPE CLASSIFICATION FROM COMMITTEE VECTORS")
print("=" * 75)

X = spec_scores_corr.T
y_type = np.array([VTYPES.index(t) for t in test_corr_types])

clf = LogisticRegression(max_iter=1000)
clf.fit(X, y_type)
y_pred = clf.predict(X)
acc = accuracy_score(y_type, y_pred)
bacc = balanced_accuracy_score(y_type, y_pred)

print(f"\n  Classifier: 6-dim specialist vector -> violation type")
print(f"  Accuracy:        {acc:.4f}")
print(f"  Balanced acc:    {bacc:.4f}")
print(f"  Chance:          {1/len(VTYPES):.4f}")
print(f"  Improvement:     {bacc/(1/len(VTYPES)):.1f}x over chance")

print(f"\n  Per-type recall:")
for i, vt in enumerate(VTYPES):
    mask = y_type == i
    recall = (y_pred[mask] == i).mean()
    print(f"    {vt:<12} {recall:.4f}  ({mask.sum()} samples)")

# Compare against baselines
X_max_only = spec_scores_corr.max(axis=0).reshape(-1, 1)
X_mean_only = spec_scores_corr.mean(axis=0).reshape(-1, 1)
X_gen_only = gen_scores_corr.reshape(-1, 1)
X_combo = np.column_stack([X_max_only, X])

clf_max = LogisticRegression(max_iter=1000)
clf_mean = LogisticRegression(max_iter=1000)
clf_gen = LogisticRegression(max_iter=1000)
clf_combo = LogisticRegression(max_iter=1000)

for name, clf_i, X_i in [
    ("Max only       ", clf_max, X_max_only),
    ("Mean only      ", clf_mean, X_mean_only),
    ("Generalist only", clf_gen, X_gen_only),
    ("Full vector    ", clf, X),
    ("Max + full vec ", clf_combo, X_combo),
]:
    clf_i.fit(X_i, y_type)
    y_p = clf_i.predict(X_i)
    b = balanced_accuracy_score(y_type, y_p)
    print(f"\n  {name} balanced acc: {b:.4f}")

# ═══════════════════════════════════════════════════════════════════════════════
# EXPERIMENT 2: Residual pattern — does disagreement add beyond max?
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 75)
print("EXPERIMENT 2: RESIDUAL PATTERN (disagreement beyond max)")
print("=" * 75)

max_valid = all_scores_valid.max(axis=0)
max_corr = all_scores_corr.max(axis=0)
mean_valid = all_scores_valid.mean(axis=0)
mean_corr = all_scores_corr.mean(axis=0)
var_valid = all_scores_valid.var(axis=0)
var_corr = all_scores_corr.var(axis=0)

residual_valid = all_scores_valid - max_valid[np.newaxis, :]
residual_corr = all_scores_corr - max_corr[np.newaxis, :]
residual_abs_valid = np.abs(residual_valid).mean(axis=0)
residual_abs_corr = np.abs(residual_corr).mean(axis=0)

cat = lambda a, b: np.concatenate([a, b])

signals = {
    "Generalist":         cat(gen_scores_valid, gen_scores_corr),
    "Mean":               cat(mean_valid, mean_corr),
    "Max":                cat(max_valid, max_corr),
    "Variance":           cat(var_valid, var_corr),
    "Residual L1":        cat(residual_abs_valid, residual_abs_corr),
}
print(f"\n{'Signal':<28} {'AUROC':>8}")
print("-" * 38)

signal_aurocs = {}
for name, sig in signals.items():
    a = roc_auc_score(y_all, sig)
    signal_aurocs[name] = a
    print(f"  {name:<26} {a:8.4f}")

print(f"\n  Variance - Max delta:   {signal_aurocs['Variance'] - signal_aurocs['Max']:+.4f}")
print(f"  Residual - Max delta:   {signal_aurocs['Residual L1'] - signal_aurocs['Max']:+.4f}")

# Logistic regression with max + residual
X_lr = np.column_stack([
    cat(max_valid, max_corr),
    cat(residual_abs_valid, residual_abs_corr),
    cat(var_valid, var_corr),
])
lr_all = LogisticRegression(max_iter=1000).fit(X_lr, y_all)
lr_max_only = LogisticRegression(max_iter=1000).fit(
    cat(max_valid, max_corr).reshape(-1, 1), y_all)

for name, lr, X_l in [
    ("Max only   ", lr_max_only, cat(max_valid, max_corr).reshape(-1, 1)),
    ("Max+Res+Var", lr_all, X_lr),
]:
    yp = lr.predict_proba(X_l)[:, 1]
    print(f"\n  LR {name} AUROC: {roc_auc_score(y_all, yp):.4f}")

# Residual type classification
print(f"\n  Residual type classification (controlling for max):")
X_res = residual_corr.T
clf_res = LogisticRegression(max_iter=1000)
clf_res.fit(X_res, y_type)
b_res = balanced_accuracy_score(y_type, clf_res.predict(X_res))
print(f"    Residual-only balanced acc: {b_res:.4f}")
print(f"    Full-vector balanced acc:   {bacc:.4f}")
print(f"    Residual retains {b_res/max(bacc, 0.001):.1%} of full-vector information")

# ═══════════════════════════════════════════════════════════════════════════════
# EXPERIMENT 3: Conditional-factorization validators (valid-only training)
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 75)
print("EXPERIMENT 3: CONDITIONAL-FACTORIZATION VALIDATORS")
print("     (trained on valid data only, tested on oracle-invalid)")
print("=" * 75)

# --- V_local: forward transition predictor ---
class VLocal(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(D_SLOT + D_ACTION, 128), nn.ReLU(),
            nn.Linear(128, D_SLOT),
        )
    def forward(self, Z, A):
        B, H_, N, d = Z.shape
        if H_ < 2: return torch.zeros(B, device=Z.device)
        z_t = Z[:, :-1]
        a_exp = A[:, :-1].unsqueeze(2).expand(-1, -1, N, -1)
        inp = torch.cat([z_t, a_exp], dim=-1)
        pred = self.net(inp)
        err = (pred - Z[:, 1:]).pow(2).mean(dim=(-2, -1)).mean(dim=-1)
        return -err

# --- V_inverse: inverse dynamics predictor ---
class VInverse(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(D_SLOT * 2, 128), nn.ReLU(),
            nn.Linear(128, D_ACTION),
        )
    def forward(self, Z, A):
        B, H_, N, d = Z.shape
        if H_ < 2: return torch.zeros(B, device=Z.device)
        z_t = Z[:, :-1].mean(dim=2)
        z_tp1 = Z[:, 1:].mean(dim=2)
        inp = torch.cat([z_t, z_tp1], dim=-1)
        pred = self.net(inp)
        err = (pred - A[:, :-1]).pow(2).mean(dim=(-2, -1))
        return -err

# --- V_slotid: slot identity consistency ---
class VSlotID(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(D_SLOT, 64)
    def forward(self, Z, A):
        z_proj = self.proj(Z)
        z_t = z_proj[:, :-1]
        z_tp1 = z_proj[:, 1:]
        sim = F.cosine_similarity(z_t, z_tp1, dim=-1).mean(dim=-1)
        return sim.diagonal(dim1=0, dim2=1).mean(dim=1) if sim.dim() > 2 else sim.mean(dim=-1)

# --- V_count: slot activity count consistency ---
class VCount(nn.Module):
    def forward(self, Z, A):
        active = (Z.norm(dim=-1) > 0.1).float()
        count = active.sum(dim=-1).float()
        count_var = count.var(dim=1)
        return -count_var

# --- V_order: temporal arrow ---
class VOrder(nn.Module):
    def __init__(self):
        super().__init__()
        self.fwd = nn.GRU(D_SLOT * N_SLOTS, 64, batch_first=True, bidirectional=True)
        self.head = nn.Sequential(nn.Linear(128, 1))
    def forward(self, Z, A):
        B = Z.shape[0]
        Zf = Z.reshape(B, H, -1)
        fwd_out, _ = self.fwd(Zf)
        rev_out, _ = self.fwd(Zf.flip(dims=[1]))
        diff = (fwd_out - rev_out.flip(dims=[1])).pow(2).mean(dim=(-2, -1))
        return -diff

# --- V_global: one-class SlotPool (like PoC validators) ---
class VGlobal(nn.Module):
    def __init__(self):
        super().__init__()
        self.pool = SlotPoolValidator()
    def forward(self, Z):
        return self.pool(Z)

# --- V_contrast: valid vs simple self-supervised corruptions ---
class VContrast(nn.Module):
    def __init__(self):
        super().__init__()
        self.cls = SlotPoolValidator()
    def forward(self, Z):
        return self.cls(Z)

# Generate self-supervised corruptions (no oracle labels)
def generate_ss_corruptions(valid_data):
    corruptions = []
    for z0, A, Z in valid_data:
        Z_c = Z.clone()
        strategy = np.random.randint(0, 5)
        if strategy == 0:
            Z_c = Z_c + torch.randn_like(Z_c) * 0.1
        elif strategy == 1:
            t = np.random.randint(0, H)
            Z_c[t:] = Z_c[t:].flip(dims=[0])
        elif strategy == 2:
            if N_SLOTS >= 2:
                i, j = np.random.choice(N_SLOTS, 2, replace=False)
                Z_c[:, i], Z_c[:, j] = Z_c[:, j].clone(), Z_c[:, i].clone()
        elif strategy == 3:
            Z_c[:, np.random.randint(0, N_SLOTS)] *= 0.1
        elif strategy == 4:
            t_split = np.random.randint(1, H)
            Z_c[t_split:] = Z_c[:H - t_split].clone()
        corruptions.append((z0.clone(), A.clone(), Z_c))
    return corruptions

# Train conditional validators
print(f"\n  V_local (forward transition):")
v_local = VLocal().to(DEVICE)
opt = torch.optim.Adam(v_local.parameters(), lr=1e-3)
for ep in range(EPOCHS):
    vi = np.random.choice(len(train_valid), BATCH, replace=False)
    Z_b = torch.stack([train_valid[i][2] for i in vi]).to(DEVICE)
    A_b = torch.stack([train_valid[i][1] for i in vi]).to(DEVICE)
    opt.zero_grad()
    err = -v_local(Z_b, A_b)
    loss = err.mean()
    loss.backward(); opt.step()
print(f"    Final reconstruction MSE: {loss.item():.4f}")

print(f"\n  V_inverse (inverse dynamics):")
v_inv = VInverse().to(DEVICE)
opt = torch.optim.Adam(v_inv.parameters(), lr=1e-3)
for ep in range(EPOCHS):
    vi = np.random.choice(len(train_valid), BATCH, replace=False)
    Z_b = torch.stack([train_valid[i][2] for i in vi]).to(DEVICE)
    A_b = torch.stack([train_valid[i][1] for i in vi]).to(DEVICE)
    opt.zero_grad()
    loss = (-v_inv(Z_b, A_b)).mean()
    loss.backward(); opt.step()
print(f"    Final inverse MSE: {loss.item():.4f}")

v_slotid = VSlotID().to(DEVICE)
v_count = VCount()
v_order = VOrder().to(DEVICE)

print(f"\n  V_global (one-class SlotPool):")
v_global = VGlobal().to(DEVICE)
opt = torch.optim.Adam(v_global.parameters(), lr=1e-3)
for ep in range(EPOCHS):
    vi = np.random.choice(len(train_valid), BATCH, replace=False)
    Z_b = torch.stack([train_valid[i][2] for i in vi]).to(DEVICE)
    opt.zero_grad()
    logits = v_global(Z_b)
    loss = F.binary_cross_entropy_with_logits(logits, torch.ones_like(logits))
    loss.backward(); opt.step()
    if (ep + 1) % 40 == 0:
        acc = (logits.sigmoid() > 0.5).float().mean()
        print(f"    ep{ep+1:3d}: loss={loss:.4f} pos_acc={acc:.3f}")

print(f"\n  V_contrast (valid vs self-supervised corruptions):")
ss_corr = generate_ss_corruptions(train_valid)
v_contrast = VContrast().to(DEVICE)
opt = torch.optim.Adam(v_contrast.parameters(), lr=1e-3)
for ep in range(EPOCHS):
    vi = np.random.choice(len(train_valid), BATCH, replace=False)
    ci = np.random.choice(len(ss_corr), BATCH, replace=False)
    Z_v = torch.stack([train_valid[i][2] for i in vi]).to(DEVICE)
    Z_c = torch.stack([ss_corr[i][2] for i in ci]).to(DEVICE)
    Z_b = torch.cat([Z_v, Z_c], dim=0)
    y = torch.cat([torch.zeros(BATCH), torch.ones(BATCH)]).to(DEVICE)
    opt.zero_grad()
    loss = F.binary_cross_entropy_with_logits(v_contrast(Z_b), y)
    loss.backward(); opt.step()
    if (ep + 1) % 40 == 0:
        acc = ((v_contrast(Z_b).sigmoid() > 0.5).float() == y).float().mean()
        print(f"    ep{ep+1:3d}: loss={loss:.4f} acc={acc:.3f}")

# Score conditional validators on test data
cond_val_names = ["local", "inverse", "slotid", "count", "order", "global", "contrast"]
cond_models = {
    "local": v_local, "inverse": v_inv, "slotid": v_slotid,
    "count": v_count, "order": v_order, "global": v_global, "contrast": v_contrast,
}

@torch.no_grad()
def score_conditional(name, model, dataset):
    scores = []
    for z0, A, Z in dataset:
        z0_d = z0.unsqueeze(0).to(DEVICE)
        A_d = A.unsqueeze(0).to(DEVICE)
        Z_d = Z.unsqueeze(0).to(DEVICE)
        if name in ("local", "inverse", "slotid", "count", "order"):
            out = model(Z_d, A_d)
        else:
            out = model(Z_d)
        scores.append(out.item())
    return np.array(scores)

# Score valid data
cond_valid_scores = {}
for name, model in cond_models.items():
    cond_valid_scores[name] = score_conditional(name, model, test_valid)

# Score corrupt data (grouped by type)
cond_corr_scores = {}
cond_corr_by_type = defaultdict(dict)
for name, model in cond_models.items():
    all_s = score_conditional(name, model, test_corr_flat)
    cond_corr_scores[name] = all_s
    for i, vt in enumerate(test_corr_types):
        cond_corr_by_type[vt][name] = cond_corr_by_type[vt].get(name, []) + [all_s[i]]

# Per-type mean scores per validator
print(f"\n{'Type':<14}", end="")
for name in cond_val_names:
    print(f" {name[:7]:>8}", end="")
print()

for vt in VTYPES:
    row = f"{vt:<14}"
    for name in cond_val_names:
        mean_score = np.mean(cond_corr_by_type[vt][name])
        row += f" {mean_score:8.4f}"
    print(row)

print(f"{'valid':<14}", end="")
for name in cond_val_names:
    print(f" {np.mean(cond_valid_scores[name]):8.4f}", end="")
print()

# Per-type AUROC for each conditional validator
print(f"\n{'Type':<14}", end="")
for name in cond_val_names:
    print(f" {name[:7]:>8}", end="")
print(f" {'best':>8}")
print("-" * (14 + 9 * len(cond_val_names)))

cond_auroc = {}
for vt in VTYPES:
    row_data = test_by_type[vt]
    n = len(row_data)
    row = f"{vt:<14}"
    best = 0
    for name in cond_val_names:
        scores_valid = cond_valid_scores[name]
        scores_corr = np.array([cond_corr_by_type[vt][name][i] for i in range(n)])
        labels = np.concatenate([np.zeros(len(scores_valid)), np.ones(n)])
        all_s = np.concatenate([scores_valid, scores_corr])
        a = roc_auc_score(labels, all_s)
        if a < 0.5: a = 1 - a  # some validators predict lower = invalid
        cond_auroc[(vt, name)] = a
        best = max(best, a)
        row += f" {a:8.4f}"
    row += f" {best:8.4f}"
    print(row)

print(f"{'MEAN':<14}", end="")
for name in cond_val_names:
    vals = [cond_auroc[(vt, name)] for vt in VTYPES]
    print(f" {np.mean(vals):8.4f}", end="")
mean_best = np.mean([max(cond_auroc[(vt, name)] for name in cond_val_names) for vt in VTYPES])
print(f" {mean_best:8.4f}")

# Overall binary AUROC per conditional validator
print(f"\n{'Validator':<14} {'AUROC':>8}")
print("-" * 24)
for name in cond_val_names:
    scores_valid = cond_valid_scores[name]
    scores_corr = cond_corr_scores[name]
    labels = np.concatenate([np.zeros(len(scores_valid)), np.ones(len(scores_corr))])
    all_s = np.concatenate([scores_valid, scores_corr])
    a = roc_auc_score(labels, all_s)
    if a < 0.5: a = 1 - a
    print(f"  {name:<12} {a:8.4f}")

# Committee combination
cond_valid_mat = np.array([cond_valid_scores[name] for name in cond_val_names])
cond_corr_mat = np.array([cond_corr_scores[name] for name in cond_val_names])

c_mean_valid = cond_valid_mat.mean(axis=0)
c_mean_corr = cond_corr_mat.mean(axis=0)
c_max_valid = cond_valid_mat.max(axis=0)
c_max_corr = cond_corr_mat.max(axis=0)
c_var_valid = cond_valid_mat.var(axis=0)
c_var_corr = cond_corr_mat.var(axis=0)

print(f"\n  Committee aggregation AUROC:")
for name, sig in [
    ("Mean  ", np.concatenate([c_mean_valid, c_mean_corr])),
    ("Max   ", np.concatenate([c_max_valid, c_max_corr])),
    ("Var   ", np.concatenate([c_var_valid, c_var_corr])),
]:
    a = roc_auc_score(y_all, sig)
    print(f"    {name}: {a:.4f}")

# ═══════════════════════════════════════════════════════════════════════════════
# Summary
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 75)
print("SUMMARY")
print("=" * 75)

print(f"""
  ORACLE-SPECIALIST COMMITTEE (supervised):
    Best single (generalist):     {signal_aurocs['Generalist']:.4f}
    Committee mean:               {signal_aurocs['Mean']:.4f}
    Committee max:                {signal_aurocs['Max']:.4f}
    Committee variance:           {signal_aurocs['Variance']:.4f}
    Variance - Max delta:         {signal_aurocs['Variance'] - signal_aurocs['Max']:+.4f}

  CONDITIONAL-FACTORIZATION COMMITTEE (unsupervised):
    Best single:                  {max(roc_auc_score(y_all, np.concatenate([cond_valid_scores[n], cond_corr_scores[n]])) for n in cond_val_names):.4f}
""")

oracle_best = signal_aurocs['Max']
cond_best = max(roc_auc_score(y_all, np.concatenate([cond_valid_scores[n], cond_corr_scores[n]])) for n in cond_val_names)
print(f"  Oracle-specialist max AUROC:    {oracle_best:.4f}")
print(f"  Conditional-factorization best: {cond_best:.4f}")
print(f"  Gap:                            {oracle_best - cond_best:.4f}")

if oracle_best - cond_best < 0.10:
    print(f"\n  VERDICT: Conditional factorizations approximate oracle specialists.")
    print(f"  True TAMG disagreement mechanism is VIABLE.")
else:
    print(f"\n  VERDICT: Conditional factorizations are weak. Oracle labels carry most signal.")
    print(f"  TAMG requires pseudo-negative specialization, not pure disagreement.")
