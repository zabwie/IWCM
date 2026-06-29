#!/usr/bin/env python3
"""IWCM vs baselines on compositional corruption grid — comparison table."""
import sys, torch, pickle, numpy as np, torch.nn.functional as F
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.seed import set_seed
from src.iwcm.slot_energy import SlotIWCMEnergy
from src.env.oracle_slot_encoder import ORACLE_SLOT_DIM, MAX_OBJECTS

set_seed(42)
device = "cuda" if torch.cuda.is_available() else "cpu"
H, N, d = 25, MAX_OBJECTS, ORACLE_SLOT_DIM
print(f"Device: {device}")

with open("data/compositional_grid.pkl", "rb") as f:
    grid = pickle.load(f)

def slot_data(entries):
    return [(torch.from_numpy(z0).float(), torch.from_numpy(A).float(),
             torch.from_numpy(Z).float()) for z0, A, Z in entries]

train_v = slot_data(grid["train_valid"])
train_c_meta = [(torch.from_numpy(enc[0]).float(), torch.from_numpy(enc[1]).float(),
                  torch.from_numpy(enc[2]).float(), meta) for enc, meta in grid["train_corr"]]
test_v = slot_data(grid["test_valid"])
test_c_meta = [(torch.from_numpy(enc[0]).float(), torch.from_numpy(enc[1]).float(),
                 torch.from_numpy(enc[2]).float(), meta) for enc, meta in grid["test_corr"]]


def evaluate_model(model, forward_fn, test_v, test_c_meta, name="model"):
    """Evaluate a model on test split, returning per-law metrics."""
    per_law = {}
    v_acc = []; i_rej = []
    for vz, vA, vZ in test_v[:80]:
        s = forward_fn(model, vz, vA, vZ)
        v_acc.append(s > 0.5)
    for cz, cA, cZ, meta in test_c_meta:
        s = forward_fn(model, cz, cA, cZ)
        i_rej.append(s < 0.5)
        law = meta["law_type"]
        per_law.setdefault(law, []).append(s < 0.5)
    law_rates = {k: np.mean(v) for k, v in per_law.items()}
    return {"valid_acc": np.mean(v_acc), "invalid_rej": np.mean(i_rej), "per_law": law_rates}


def train_model(model, forward_fn, train_v, train_c_meta, name, epochs=150, lr=3e-4):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    MARGIN, REG = 1.0, 0.001
    model.to(device)
    for ep in range(epochs):
        vi = np.random.choice(len(train_v), min(16, len(train_v)), replace=False)
        ci = np.random.choice(len(train_c_meta), min(32, len(train_c_meta)), replace=False)
        vz0 = torch.stack([train_v[i][0] for i in vi]).to(device)
        vA = torch.stack([train_v[i][1] for i in vi]).to(device)
        vZ = torch.stack([train_v[i][2] for i in vi]).to(device)
        cz0 = torch.stack([train_c_meta[i][0] for i in ci]).to(device)
        cA = torch.stack([train_c_meta[i][1] for i in ci]).to(device)
        cZ = torch.stack([train_c_meta[i][2] for i in ci]).to(device)
        opt.zero_grad()
        ev = model(vz0, vA, vZ); ec = model(cz0, cA, cZ)
        loss = F.relu(ev + MARGIN).mean() + F.relu(MARGIN - ec).mean() + REG * (ev.pow(2).mean() + ec.pow(2).mean())
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
    return evaluate_model(model, forward_fn, test_v, test_c_meta, name)


# Baseline 1: Flat MLP classifier
print("\n=== MLP Classifier ===")
class FlatMLP(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(H * N * d, 256), torch.nn.ReLU(),
            torch.nn.Linear(256, 64), torch.nn.ReLU(),
            torch.nn.Linear(64, 1))
    def forward(self, z0, A, Z): return self.net(Z.reshape(Z.shape[0], -1)).squeeze(-1)

def mlp_forward(model, z0, A, Z):
    return torch.sigmoid(-model(z0.unsqueeze(0).to(device), A.unsqueeze(0).to(device), Z.unsqueeze(0).to(device))).item()

mlp = FlatMLP()
mlp_r = train_model(mlp, mlp_forward, train_v, train_c_meta, "MLP")
print(f"MLP: {mlp_r}")

# Baseline 2: Slot Transformer
print("\n=== Slot Transformer ===")
class SlotTransformer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = torch.nn.Linear(d, 64)
        enc = torch.nn.TransformerEncoderLayer(d_model=64, nhead=4, dim_feedforward=128, dropout=0.1, batch_first=True)
        self.transformer = torch.nn.TransformerEncoder(enc, num_layers=2)
        self.scorer = torch.nn.Linear(64, 1)
    def forward(self, z0, A, Z):
        B = Z.shape[0]; Zp = self.proj(Z).reshape(B, H * N, 64)
        Ze = self.transformer(Zp).mean(dim=1)
        return self.scorer(Ze).squeeze(-1)

st = SlotTransformer()
st_r = train_model(st, mlp_forward, train_v, train_c_meta, "SlotTransformer", epochs=150)
print(f"SlotTransformer: {st_r}")

# Baseline 3: IWCM Slot-Aware
print("\n=== IWCM Slot-Aware ===")
iwcm = SlotIWCMEnergy(d_slot=d, d_action=11, hidden_dim=128, num_slots=N)
def iwcm_forward(model, z0, A, Z):
    return model.score_acceptance(z0.unsqueeze(0).to(device), A.unsqueeze(0).to(device), Z.unsqueeze(0).to(device)).item()

iwcm_r = train_model(iwcm, iwcm_forward, train_v, train_c_meta, "IWCM", epochs=200, lr=3e-4)
print(f"IWCM: {iwcm_r}")

# Comparison Table
print("\n" + "=" * 80)
print("COMPARISON TABLE")
print("=" * 80)
print(f"{'Model':<25} {'Valid Acc':>10} {'Invalid Rej':>12} {'Conservation':>14} {'Identity':>14}")
print("-" * 80)
for name, r in [("MLP Classifier", mlp_r), ("Slot Transformer", st_r), ("IWCM Slot-Aware", iwcm_r)]:
    cons = r["per_law"].get("conservation", 0.0)
    ident = r["per_law"].get("identity", 0.0)
    print(f"{name:<25} {r['valid_acc']:>10.3f} {r['invalid_rej']:>12.3f} {cons:>14.3f} {ident:>14.3f}")
print("=" * 80)
torch.save({"mlp": mlp.state_dict(), "st": st.state_dict(), "iwcm": iwcm.state_dict()},
           "outputs/checkpoints/comparison_models.pt")
print("Models saved.")
