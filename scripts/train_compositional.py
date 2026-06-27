#!/usr/bin/env python3
"""Train slot-aware IWCM on compositional corruption grid — with shortcut audit."""
import sys, torch, pickle, numpy as np, torch.nn.functional as F
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.seed import set_seed
from src.iwcm.slot_energy import SlotIWCMEnergy
from src.encoder.oracle_slot_encoder import ORACLE_SLOT_DIM, MAX_OBJECTS

set_seed(42)
device = "cuda" if torch.cuda.is_available() else "cpu"
H, N, d = 25, MAX_OBJECTS, ORACLE_SLOT_DIM
D_SLOT = N * d  # 152
print(f"Device: {device}, slots={N}x{d}={D_SLOT}")

# Load compositional grid
with open("data/compositional_grid.pkl", "rb") as f:
    grid = pickle.load(f)

# Shortcut classifier audit: can metadata alone predict valid/invalid?
print("\n=== Shortcut Audit ===")
meta_feats = []
meta_labels = []
# Shortcut classifier audit: use one-hot categorical encoding
object_types = ["key", "box"]
contexts = ["visible", "occluded", "carried"]
vtypes = ["duplicate", "delete", "transform", "swap", "teleport", "illegal_open", "reverse"]
time_gaps = ["early", "mid", "late"]
laws = ["conservation", "identity", "locality", "temporal"]

meta_feats = []
meta_labels = []
rng = np.random.RandomState(42)
for _, meta in grid["train_corr"][:200]:
    # One-hot encode each category — randomized for valid entries
    feat = []
    feat.extend([1.0 if meta["object_type"] == ot else 0.0 for ot in object_types])
    feat.extend([1.0 if meta["context"] == ct else 0.0 for ct in contexts])
    feat.extend([1.0 if meta["violation_type"] == vt else 0.0 for vt in vtypes])
    feat.extend([1.0 if meta["time_gap"] == tg else 0.0 for tg in time_gaps])
    feat.extend([1.0 if meta["law_type"] == lt else 0.0 for lt in laws])
    meta_feats.append(feat); meta_labels.append(1)
for _ in range(200):
    feat = [rng.random() for _ in range(len(object_types)+len(contexts)+len(vtypes)+len(time_gaps)+len(laws))]
    meta_feats.append(feat); meta_labels.append(0)

XF = torch.tensor(meta_feats, dtype=torch.float32)
YF = torch.tensor(meta_labels, dtype=torch.float32)
n_feats = XF.shape[1]
sc = torch.nn.Sequential(torch.nn.Linear(n_feats, 32), torch.nn.ReLU(), torch.nn.Linear(32, 1))
opt_sc = torch.optim.Adam(sc.parameters(), lr=1e-3)
for _ in range(200):
    opt_sc.zero_grad()
    loss_sc = F.binary_cross_entropy_with_logits(sc(XF).squeeze(), YF)
    loss_sc.backward(); opt_sc.step()
with torch.no_grad():
    preds = (sc(XF).squeeze() > 0).float()
    sc_acc = (preds == YF).float().mean().item()
print(f"Shortcut classifier accuracy: {sc_acc:.4f} (goal: near 0.500)")
print(f"SHORTCUT IMMUNITY: {'PASS' if abs(sc_acc-0.5) < 0.1 else 'FAIL - data has shortcuts'}")

# Flat IWCM training on compositional grid
print("\n=== Training Flat IWCM on Compositional Grid ===")
D_FLAT = 8 * 8 * 4  # 256

# Encode valid trajectories as oracle slots (keep 4D structure)
valid_slots = []
for z0f, Af, Zf in grid["train_valid"]:
    valid_slots.append((
        torch.from_numpy(z0f).float(),
        torch.from_numpy(Af).float(),
        torch.from_numpy(Zf).float(),
    ))

# Encode corruptions as oracle slots
corr_slots_meta = []
for (z0f, Af, Zf), meta in grid["train_corr"]:
    corr_slots_meta.append((
        torch.from_numpy(z0f).float(),
        torch.from_numpy(Af).float(),
        torch.from_numpy(Zf).float(),
        meta,
    ))

model = SlotIWCMEnergy(d_slot=d, d_action=11, hidden_dim=128, num_slots=N).to(device)
opt = torch.optim.Adam(model.parameters(), lr=3e-4)
print(f"Model: {sum(p.numel() for p in model.parameters()):,} params")

MARGIN, REG = 1.0, 0.001
NUM_EPOCHS = 200
BATCH = 16

for epoch in range(NUM_EPOCHS):
    vi = np.random.choice(len(valid_slots), min(BATCH, len(valid_slots)), replace=False)
    ci = np.random.choice(len(corr_slots_meta), min(BATCH * 2, len(corr_slots_meta)), replace=False)

    vz0 = torch.stack([valid_slots[i][0] for i in vi]).to(device)
    vA = torch.stack([valid_slots[i][1] for i in vi]).to(device)
    vZ = torch.stack([valid_slots[i][2] for i in vi]).to(device)
    cz0 = torch.stack([corr_slots_meta[i][0] for i in ci]).to(device)
    cA = torch.stack([corr_slots_meta[i][1] for i in ci]).to(device)
    cZ = torch.stack([corr_slots_meta[i][2] for i in ci]).to(device)

    opt.zero_grad()
    ev = model(vz0, vA, vZ); ec = model(cz0, cA, cZ)
    loss = F.relu(ev + MARGIN).mean() + F.relu(MARGIN - ec).mean() + REG * (ev.pow(2).mean() + ec.pow(2).mean())
    loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()

    if (epoch + 1) % 20 == 0:
        print(f"Epoch {epoch+1}: loss={loss.item():.4f} E_valid={ev.mean().item():.3f} "
              f"E_invalid={ec.mean().item():.3f} gap={ec.mean().item()-ev.mean().item():.3f}")

# Evaluate on test split
print("\n=== Cross-Surface Evaluation (compositional test split) ===")
model.eval()

# Encode test valid as slots
test_valid_s = [(torch.from_numpy(z0).float(), torch.from_numpy(A).float(),
                 torch.from_numpy(Z).float()) for z0, A, Z in grid["test_valid"]]

# Encode test corruptions as slots
test_corr_s = [(torch.from_numpy(enc[0]).float(), torch.from_numpy(enc[1]).float(),
                torch.from_numpy(enc[2]).float(), meta) for enc, meta in grid["test_corr"]]

ev_list = []
for vz, vA, vZ in test_valid_s[:50]:
    s = model.score_acceptance(vz.to(device).unsqueeze(0), vA.to(device).unsqueeze(0), vZ.to(device).unsqueeze(0)).item()
    ev_list.append(s)
ec_list = []
for cz, cA, cZ, meta in test_corr_s[:100]:
    s = model.score_acceptance(cz.to(device).unsqueeze(0), cA.to(device).unsqueeze(0), cZ.to(device).unsqueeze(0)).item()
    ec_list.append(s)

print(f"Valid accept: mean={np.mean(ev_list):.4f} >0.5:{(np.array(ev_list)>0.5).mean():.3f}")
print(f"Invalid accept: mean={np.mean(ec_list):.4f} <0.5:{(np.array(ec_list)<0.5).mean():.3f}")

# Per-law breakdown from test meta
per_law_scores = {}
for cz, cA, cZ, meta in test_corr_s:
    law = meta["law_type"]
    s = model.score_acceptance(cz.to(device).unsqueeze(0), cA.to(device).unsqueeze(0), cZ.to(device).unsqueeze(0)).item()
    per_law_scores.setdefault(law, []).append(s)

print("\nPer-law rejection rate:")
for law, scores in per_law_scores.items():
    print(f"  {law}: mean_accept={np.mean(scores):.4f} rejected={(np.array(scores)<0.5).mean():.3f} (N={len(scores)})")

torch.save(model.state_dict(), "outputs/checkpoints/iwcm_compositional_grid.pt")
print("\nSaved to outputs/checkpoints/iwcm_compositional_grid.pt")
