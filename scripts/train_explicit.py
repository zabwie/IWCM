#!/usr/bin/env python3
"""Train IWCM directly on explicit valid+corruption pairs for cross-surface generalization."""
import sys, torch, pickle, numpy as np, torch.nn.functional as F
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.seed import set_seed
from src.iwcm.model import IWCM
from src.metrics.evaluation import metric_cross_surface_law_generalization, metric_valid_invalid_classification

set_seed(42)
device = "cuda" if torch.cuda.is_available() else "cpu"
GRID, D_STATE, D_ACTION, H = 8, 8*8*4, 11, 25
print(f"Device: {device}")

# Load data
with open("data/corruption_train.pkl", "rb") as f:
    cdata = pickle.load(f)
with open("data/cross_surface.pkl", "rb") as f:
    evdata = pickle.load(f)

def np_to_torch(trajs):
    return [(torch.from_numpy(z0).reshape(-1), torch.from_numpy(A),
             torch.from_numpy(Z).reshape(Z.shape[0], -1)) for z0, A, Z in trajs]

valid = np_to_torch(cdata["valid"])
corruptions = {}
all_corr = []
for law in ["conservation", "identity", "locality", "temporal"]:
    tr = np_to_torch(cdata["corruptions"][law]["train"])
    corruptions[law] = tr
    all_corr.extend(tr)

# Eval data
ev_valid = np_to_torch(evdata.get("valid", [])[:200])
ev_train = {k: {"invalid": np_to_torch(v)} for k, v in evdata["train"].items()}
ev_test = {k: {"invalid": np_to_torch(v)} for k, v in evdata["test"].items()}
con_inv = np_to_torch(evdata["train"].get("conservation", [])[:100])

print(f"Train: {len(valid)} valid, {len(all_corr)} corrupted")
print(f"Corruptions: " + ", ".join(f"{k}={len(v)}" for k, v in corruptions.items()))

# Model
model = IWCM(d_state=D_STATE, d_action=D_ACTION, hidden_dim=384)
opt = torch.optim.Adam(model.parameters(), lr=5e-5)
model.to(device)
print(f"Model: {model.count_parameters_str()} params")

NUM_EPOCHS = 200
BATCH = 32
MARGIN = 2.0
REG = 0.0001
best_ho = 0.0

for epoch in range(NUM_EPOCHS):
    # Shuffle
    idx = np.random.permutation(len(valid))[:BATCH]
    v_batch = [valid[i] for i in idx]
    # Sample corruptions
    c_batch = []
    for law in corruptions:
        if corruptions[law]:
            n_sample = min(BATCH//2, len(corruptions[law]))
            ci = np.random.choice(len(corruptions[law]), n_sample, replace=False)
            c_batch.extend([corruptions[law][j] for j in ci])

    v_z0 = torch.stack([x[0] for x in v_batch]).to(device)
    v_A = torch.stack([x[1] for x in v_batch]).to(device)
    v_Z = torch.stack([x[2] for x in v_batch]).to(device)
    c_z0 = torch.stack([x[0] for x in c_batch]).to(device)
    c_A = torch.stack([x[1] for x in c_batch]).to(device)
    c_Z = torch.stack([x[2] for x in c_batch]).to(device)

    opt.zero_grad()
    ev = model.energy(v_z0, v_A, v_Z)
    ec = model.energy(c_z0, c_A, c_Z)
    loss = F.relu(ev + MARGIN).mean() + F.relu(MARGIN - ec).mean() + REG * (ev.pow(2).mean() + ec.pow(2).mean())
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()

    if (epoch+1) % 20 == 0:
        print(f"Epoch {epoch+1}: loss={loss.item():.4f} E_valid={ev.mean().item():.2f} E_invalid={ec.mean().item():.2f}")

    if (epoch+1) % 50 == 0:
        model.eval()
        csr = metric_cross_surface_law_generalization(model, ev_train, ev_test, device)
        ho = csr.get("held_out_accuracy", 0.0)
        print(f"  Cross-surface: ID={csr.get('in_distribution_accuracy',0):.3f} HeldOut={ho:.3f} Gap={csr.get('generalization_gap',0):.3f}")
        if ho > best_ho:
            best_ho = ho
            model.save("outputs/checkpoints/iwcm_explicit_cs.pt")

model.save("outputs/checkpoints/iwcm_explicit_final.pt")

print(f"\nFinal eval...")
model.eval()
csr = metric_cross_surface_law_generalization(model, ev_train, ev_test, device)
print(f"Cross-surface: ID={csr['in_distribution_accuracy']:.3f} HeldOut={csr['held_out_accuracy']:.3f}")
for law, v in csr["per_law_breakdown"].items():
    print(f"  {law}: valid_acc={v['valid_accuracy']:.3f} invalid_rej={v['invalid_rejection']:.3f}")

cls = metric_valid_invalid_classification(model, ev_valid, {"conservation": con_inv}, device)
print(f"Classification: AUROC={cls['AUROC']:.3f} FPR={cls['FPR']:.3f}")
print(f"Best held-out: {best_ho:.3f}")
