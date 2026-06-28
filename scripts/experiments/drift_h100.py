#!/usr/bin/env python3
"""Drift at H=100: rollout vs IWCM warm-start. Single script, hardcoded paths.
   ponytail: one-shot experiment, not reusable."""
import sys, pickle, torch, numpy as np
from pathlib import Path
BASE = str(Path(__file__).resolve().parent.parent.parent)
sys.path.insert(0, BASE); sys.path.insert(0, BASE + '/src')
torch.set_num_threads(1)
import torch.nn as nn
from src.iwcm.slot_energy import SlotIWCMEnergy
from src.env.scenarios import generate_trajectory, Scenario, PREDEFINED_SCENARIOS
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
N, d_slot, H, d_a = 8, 19, 100, 11

# ── 1. Generate H=100 trajectories from grid world ──────────
print("Generating H=100 trajectories...")
# Use all scenarios for diverse training data
all_trajs = []
for name in PREDEFINED_SCENARIOS:
    scenario = Scenario.from_preset(name, 8)
    for s in range(80):
        states, actions, _ = generate_trajectory(scenario, H, policy='mixed', seed=42+s)
        if len(states) >= H+1:
            all_trajs.append((states, actions))
print(f"  {len(all_trajs)} valid trajectories")

# Encode to oracle slots
import importlib.util
spec = importlib.util.spec_from_file_location('ose', BASE + '/src/encoder/oracle_slot_encoder.py')
ose = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ose)
encoded = [ose.encode_oracle_trajectory(s, a, horizon=H) for s, a in all_trajs]
encoded = [e for e in encoded if e is not None]
print(f"  {len(encoded)} after oracle encoding")

np.random.seed(42)
np.random.shuffle(encoded)
split = int(len(encoded) * 0.8)
train_data = [(torch.from_numpy(z0).float(), torch.from_numpy(A).float(),
               torch.from_numpy(Z).float()) for z0, A, Z in encoded[:split]]
test_data  = [(torch.from_numpy(z0).float(), torch.from_numpy(A).float(),
               torch.from_numpy(Z).float()) for z0, A, Z in encoded[split:]]
print(f"  Train: {len(train_data)}, Test: {len(test_data)}")

# ── 2. Build + train rollout model ──────────────────────────
class RolloutMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(N*d_slot + d_a, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, N*d_slot),
        )
    def forward(self, z, a):
        B = z.shape[0]
        return self.net(torch.cat([z.reshape(B, -1), a], dim=-1)).reshape(B, N, d_slot)
    def rollout(self, z0, A):
        B, Hf = A.shape[:2]
        z = z0; zs = []
        for t in range(Hf):
            z = self(z, A[:, t]); zs.append(z)
        return torch.stack(zs, dim=1)

model_rl = RolloutMLP().to(DEVICE)
opt = torch.optim.Adam(model_rl.parameters(), lr=1e-3)

all_pairs = []
for z0, A, Z in train_data:
    Z_full = torch.cat([z0.unsqueeze(0), Z], dim=0)
    for t in range(H):
        all_pairs.append((Z_full[t], A[t], Z_full[t+1]))
n_pairs = len(all_pairs)
print(f"Training rollout model on {n_pairs} transition pairs...")

BATCH = 512
for ep in range(100):
    idxs = torch.randperm(n_pairs)
    losses = []
    for i in range(0, n_pairs, BATCH):
        idx = idxs[i:i+BATCH]
        zs = torch.stack([all_pairs[j][0] for j in idx]).to(DEVICE)
        ac = torch.stack([all_pairs[j][1] for j in idx]).to(DEVICE)
        nxt = torch.stack([all_pairs[j][2] for j in idx]).to(DEVICE)
        loss = nn.MSELoss()(model_rl(zs, ac), nxt)
        opt.zero_grad(); loss.backward(); opt.step()
        losses.append(loss.item())
    if (ep+1) % 20 == 0:
        print(f"  ep {ep+1:3d}: loss={np.mean(losses):.6f}")

# ── 3. Load pretrained IWCM ─────────────────────────────────
print("Loading IWCM...")
iwcm = SlotIWCMEnergy(d_slot=d_slot, d_action=d_a, hidden_dim=192, num_slots=N).to(DEVICE)
iwcm.load_state_dict(torch.load("outputs/checkpoints/slot_iwcm_energy_v2_perobject.pt", map_location=DEVICE, weights_only=True))
iwcm.eval()

# ── 4. Solver (inline GD) ───────────────────────────────────
def solve_slots(z0, A, steps=50, lr=0.01, init_Z=None):
    iwcm.train()
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

# ── 5. Evaluate drift at multiple horizons ──────────────────
# Measure at every 10th step to show the divergence curve
measure_steps = list(range(10, H+1, 10))
results = {t: [] for t in measure_steps}
n_test = min(len(test_data), 100)

for idx in range(n_test):
    if (idx+1) % 20 == 0:
        print(f"  evaluating {idx+1}/{n_test}...")
    z0, A, Z_true = test_data[idx]
    z0_b = z0.unsqueeze(0).to(DEVICE)
    A_b = A.unsqueeze(0).to(DEVICE)
    Z_true_b = Z_true.unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        Z_rollout = model_rl.rollout(z0_b, A_b)
    Z_warm = solve_slots(z0_b, A_b, init_Z=Z_rollout, steps=100, lr=0.005)

    for t in measure_steps:
        with torch.no_grad():
            e_true = iwcm(z0_b, A_b[:, :t], Z_true_b[:, :t]).item()
            e_roll = iwcm(z0_b, A_b[:, :t], Z_rollout[:, :t]).item()
            e_warm = iwcm(z0_b, A_b[:, :t], Z_warm[:, :t]).item()
        results[t].append((e_true, e_roll, e_warm))

# ── 6. Print table ──────────────────────────────────────────
print("\n" + "=" * 80)
print(f"{'Step':>6} | {'E(true)':>9} {'E(roll)':>9} {'E(warm)':>9} | {'Δ roll':>9} {'Δ warm':>9}")
print("-" * 80)
for t in measure_steps:
    v = results[t]
    e_true = np.mean([x[0] for x in v])
    e_roll = np.mean([x[1] for x in v])
    e_warm = np.mean([x[2] for x in v])
    print(f"  {t:4d} | {e_true:9.3f} {e_roll:9.3f} {e_warm:9.3f} | "
          f"{e_roll-e_true:+9.3f} {e_warm-e_true:+9.3f}")

# ── 7. CSV for plotting ─────────────────────────────────────
print("\n\nCSV for plotting (copy into any plotting tool):")
print("step,true_energy,rollout_energy,warm_energy,rollout_drift,warm_drift")
for t in measure_steps:
    v = results[t]
    e_true = np.mean([x[0] for x in v])
    e_roll = np.mean([x[1] for x in v])
    e_warm = np.mean([x[2] for x in v])
    print(f"{t},{e_true:.4f},{e_roll:.4f},{e_warm:.4f},{e_roll-e_true:+.4f},{e_warm-e_true:+.4f}")

# Summary
print("\n" + "=" * 80)
dr_final = np.mean([results[100][i][1] - results[100][i][0] for i in range(len(results[100]))])
dw_final = np.mean([results[100][i][2] - results[100][i][0] for i in range(len(results[100]))])
print(f"At H=100: rollout Δ={dr_final:+.3f}, warm-start IWCM Δ={dw_final:+.3f}")
print(f"IWCM eliminates {((1 - dw_final/dr_final)*100):.0f}% of autoregressive drift."
      if dw_final < dr_final else "IWCM drift exceeds rollout drift.")
print("=" * 80)
