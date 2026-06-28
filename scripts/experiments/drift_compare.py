#!/usr/bin/env python3
"""Drift comparison: rollout model vs IWCM solver on causal violation rate.
   ponytail: one-shot experiment, not a library module. Hardcoded paths."""
import sys, pickle, torch, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
torch.set_num_threads(1)  # fast enough without dataloader parallelism
import torch.nn as nn
from src.iwcm.slot_energy import SlotIWCMEnergy
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
N, d_slot, H, d_action = 8, 19, 25, 11  # ponytail: constants, not config

# ── 1. Load data ────────────────────────────────────────────
with open("data/compositional_grid.pkl", "rb") as f:
    data = pickle.load(f)
train_valid = [(torch.from_numpy(z0).float(), torch.from_numpy(A).float(),
                torch.from_numpy(Z).float()) for z0, A, Z in data["train_valid"]]
test_valid  = [(torch.from_numpy(z0).float(), torch.from_numpy(A).float(),
                torch.from_numpy(Z).float()) for z0, A, Z in data["test_valid"]]
print(f"Train: {len(train_valid)} valid trajectories, Test: {len(test_valid)} valid trajectories")

# ── 2. Build + train rollout model ──────────────────────────
class RolloutMLP(nn.Module):
    """Learned transition f(z_t, a_t) → z_{t+1}. Autoregressive at inference."""
    def __init__(self):
        super().__init__()
        d_in = N * d_slot + d_action
        self.net = nn.Sequential(
            nn.Linear(d_in, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, N * d_slot),
        )
    def forward(self, z, a):  # z: (B, N, d_slot), a: (B, d_action)
        B = z.shape[0]
        return self.net(torch.cat([z.reshape(B, -1), a], dim=-1)).reshape(B, N, d_slot)
    def rollout(self, z0, A):  # A: (B, H, d_action) → (B, H, N, d_slot)
        B, Hf = A.shape[:2]
        z = z0
        zs = []
        for t in range(Hf):
            z = self(z, A[:, t])
            zs.append(z)
        return torch.stack(zs, dim=1)

model_rl = RolloutMLP().to(DEVICE)
opt = torch.optim.Adam(model_rl.parameters(), lr=1e-3)

# Extract (z_t, a_t, z_{t+1}) pairs — one big flat list, shuffle, batch
all_pairs = []
for z0, A, Z in train_valid:
    Z_full = torch.cat([z0.unsqueeze(0), Z], dim=0)  # (H+1, N, d_slot)
    for t in range(H):
        all_pairs.append((Z_full[t], A[t], Z_full[t+1]))

print(f"Training rollout model on {len(all_pairs)} transition pairs...")
BATCH = 256
for ep in range(50):
    idxs = torch.randperm(len(all_pairs))
    losses = []
    for i in range(0, len(idxs), BATCH):
        idx = idxs[i:i+BATCH]
        zs = torch.stack([all_pairs[j][0] for j in idx]).to(DEVICE)
        ac = torch.stack([all_pairs[j][1] for j in idx]).to(DEVICE)
        nxt = torch.stack([all_pairs[j][2] for j in idx]).to(DEVICE)
        pred = model_rl(zs, ac)
        loss = nn.MSELoss()(pred, nxt)
        opt.zero_grad(); loss.backward(); opt.step()
        losses.append(loss.item())
    if (ep+1) % 10 == 0:
        print(f"  ep {ep+1:3d}: loss={np.mean(losses):.6f}")

# ── 3. Load pretrained IWCM ─────────────────────────────────
print("Loading pretrained IWCM (slot_iwcm_energy_v2.pt)...")
iwcm = SlotIWCMEnergy(d_slot=d_slot, d_action=d_action, hidden_dim=192, num_slots=N).to(DEVICE)
iwcm.load_state_dict(torch.load("outputs/checkpoints/slot_iwcm_energy_v2_perobject.pt", map_location=DEVICE, weights_only=True))
iwcm.eval()

# ── 4. Inline GD solver (avoids solver.py d_state mismatch with slot format) ──
# ponytail: 10-line loop, cleaner than wrapping existing code
def solve_slots(z0, A, steps=50, lr=0.01, init_Z=None):
    # Z^{(k+1)} = Z^{(k)} - α·∇_Z E_θ
    iwcm.train()  # ponytail: CuDNN RNN backward needs train mode; no params updated
    B, Hf = A.shape[:2]
    Z = (init_Z.clone().detach() if init_Z is not None
         else torch.randn(B, Hf, N, d_slot, device=DEVICE))
    Z.requires_grad_(True)
    vel = torch.zeros_like(Z)
    for _ in range(steps):
        e = iwcm(z0, A, Z).mean()
        g = torch.autograd.grad(e, Z, create_graph=False)[0]
        vel = 0.9 * vel + g
        Z = Z.detach() - lr * vel
        Z.requires_grad_(True); vel = vel.detach()
    iwcm.eval()
    return Z

# ── 5. Evaluate: per-step energy (drift indicator) ──────────
# ponytail: energy = drift metric — lower means fewer causal violations
test_steps = [5, 10, 15, 20, 25]
results = {t: [] for t in test_steps}

for idx, (z0, A, Z_true) in enumerate(test_valid):
    if (idx+1) % 50 == 0:
        print(f"  evaluating {idx+1}/{len(test_valid)}...")
    z0_b = z0.unsqueeze(0).to(DEVICE)
    A_b = A.unsqueeze(0).to(DEVICE)
    Z_true_b = Z_true.unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        Z_rollout = model_rl.rollout(z0_b, A_b)
    Z_iwcm = solve_slots(z0_b, A_b)
    Z_warm = solve_slots(z0_b, A_b, init_Z=Z_rollout)

    for t in test_steps:
        with torch.no_grad():
            e_true = iwcm(z0_b, A_b[:, :t], Z_true_b[:, :t]).item()
            e_roll = iwcm(z0_b, A_b[:, :t], Z_rollout[:, :t]).item()
            e_iwcm = iwcm(z0_b, A_b[:, :t], Z_iwcm[:, :t]).item()
            e_warm = iwcm(z0_b, A_b[:, :t], Z_warm[:, :t]).item()
        results[t].append((e_true, e_roll, e_iwcm, e_warm))

# ── 6. Print table ──────────────────────────────────────────
print("\n" + "=" * 100)
hdr = f"{'Step':>6} | {'E(true)':>9} {'E(roll)':>9} {'E(iwcm)':>9} {'E(warm)':>9} | {'Δ roll':>9} {'Δ iwcm':>9} {'Δ warm':>9}"
print(hdr)
print("-" * 100)
for t in test_steps:
    v = results[t]
    e_true = np.mean([x[0] for x in v])
    e_roll = np.mean([x[1] for x in v])
    e_iwcm = np.mean([x[2] for x in v])
    e_warm = np.mean([x[3] for x in v])
    print(f"  {t:4d} | {e_true:9.3f} {e_roll:9.3f} {e_iwcm:9.3f} {e_warm:9.3f} | "
          f"{e_roll-e_true:+9.3f} {e_iwcm-e_true:+9.3f} {e_warm-e_true:+9.3f}")

print("-" * 100)
dr = np.mean([results[25][i][1] - results[25][i][0] for i in range(len(results[25]))])
di = np.mean([results[25][i][2] - results[25][i][0] for i in range(len(results[25]))])
dw = np.mean([results[25][i][3] - results[25][i][0] for i in range(len(results[25]))])
print(f"At step 25 — rollout Δ={dr:+.3f}, IWCM Δ={di:+.3f}, warm-start Δ={dw:+.3f}")
print("=" * 100)
