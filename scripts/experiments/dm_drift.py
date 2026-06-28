#!/usr/bin/env python3
"""DM Control drift at H=100: rollout vs IWCM warm-start on real physics.
   ponytail: one-shot experiment, not reusable."""
import sys, time, torch, numpy as np
from pathlib import Path
BASE = str(Path(__file__).resolve().parent.parent)
sys.path.insert(0, BASE); sys.path.insert(0, BASE + '/src')
torch.set_num_threads(1)
import torch.nn as nn, torch.nn.functional as F
from src.env.dm_control_wrapper import DMControlWrapper
from src.env.dm_control_encoder import DMControlOracleEncoder, MAX_BODIES, ORACLE_SLOT_DIM
from src.iwcm.fused_energy import FusedIWCMEnergy
from src.utils.seed import set_seed
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
H = 100; N_slots = MAX_BODIES; d_slot = ORACLE_SLOT_DIM  # 8, 19

print(f"DM Control drift experiment (H={H}, device={DEVICE})")
set_seed(42); rng = np.random.RandomState(42)

# ── 1. Generate data ────────────────────────────────────────
wrapper = DMControlWrapper('cartpole', 'swingup', seed=42, max_episode_steps=300)
encoder = DMControlOracleEncoder('cartpole')
d_action = wrapper.action_dim
print(f"Action dim: {d_action}, Bodies: cart, pole")

def gen_traj(horizon=H, corrupt=None):
    if corrupt:
        result = wrapper.generate_corrupted_trajectory(horizon, corruption_type=corrupt, rng=rng)
    else:
        result = wrapper.generate_trajectory(horizon, random_policy=True)
    if result is None: return None
    if corrupt:
        physics_states, _, actions, meta = result
    else:
        physics_states, _, actions = result
    enc = encoder.encode_trajectory(physics_states, actions, horizon)
    if enc is None: return None
    z0, A, Z = enc
    return (torch.from_numpy(z0).float(), torch.from_numpy(A).float(),
            torch.from_numpy(Z).float())

print("Generating trajectories...")
t0 = time.time()
train_valid = [gen_traj() for _ in range(400)]
train_valid = [t for t in train_valid if t is not None]
corrupt_types = ['teleport', 'freeze', 'reverse']
train_corrupt = []
for ct in corrupt_types:
    for _ in range(100):
        t = gen_traj(corrupt=ct)
        if t: train_corrupt.append(t)
train_corrupt = train_corrupt[:200]
test_valid = [gen_traj() for _ in range(100)]
test_valid = [t for t in test_valid if t is not None]
print(f"  {len(train_valid)} valid + {len(train_corrupt)} corrupt train, {len(test_valid)} test ({time.time()-t0:.1f}s)")

# ── 2. Train IWCM energy function ──────────────────────────
print(f"Training IWCM (FusedIWCMEnergy, 52K params)...")
set_seed(42)
iwcm = FusedIWCMEnergy(d_slot=d_slot, d_action=d_action, hidden=128, num_slots=N_slots).to(DEVICE)
opt = torch.optim.Adam(iwcm.parameters(), lr=1e-3)
BATCH = 64
t0 = time.time()
for ep in range(200):
    vi = np.random.choice(len(train_valid), min(BATCH, len(train_valid)), replace=False)
    ci = np.random.choice(len(train_corrupt), min(BATCH, len(train_corrupt)), replace=False)
    vz0 = torch.stack([train_valid[i][0] for i in vi]).to(DEVICE)
    vA = torch.stack([train_valid[i][1] for i in vi]).to(DEVICE)
    vZ = torch.stack([train_valid[i][2] for i in vi]).to(DEVICE)
    cz0 = torch.stack([train_corrupt[i][0] for i in ci]).to(DEVICE)
    cA = torch.stack([train_corrupt[i][1] for i in ci]).to(DEVICE)
    cZ = torch.stack([train_corrupt[i][2] for i in ci]).to(DEVICE)
    ev = iwcm(vz0, vA, vZ); ec = iwcm(cz0, cA, cZ)
    loss = F.relu(ev + 1.0).mean() + F.relu(1.0 - ec).mean() + 0.001*(ev.pow(2).mean()+ec.pow(2).mean())
    opt.zero_grad(); loss.backward(); opt.step()
    if (ep+1) % 50 == 0:
        print(f"  ep {ep+1:3d}: ev={ev.mean().item():+.3f} ec={ec.mean().item():+.3f} loss={loss.item():.4f}")
print(f"  Done ({time.time()-t0:.1f}s)")
iwcm.eval()

# ── 3. Train rollout model ──────────────────────────────────
print(f"Training rollout model on {len(train_valid)} trajectories...")
class RolloutMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(N_slots*d_slot + d_action, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, N_slots*d_slot),
        )
    def forward(self, z, a):
        B = z.shape[0]
        return self.net(torch.cat([z.reshape(B, -1), a], dim=-1)).reshape(B, N_slots, d_slot)
    def rollout(self, z0, A):
        B, Hf = A.shape[:2]; z = z0; zs = []
        for t in range(Hf): z = self(z, A[:, t]); zs.append(z)
        return torch.stack(zs, dim=1)

rl = RolloutMLP().to(DEVICE); opt_rl = torch.optim.Adam(rl.parameters(), lr=1e-3)
all_pairs = []
for z0, A, Z in train_valid:
    Zf = torch.cat([z0.unsqueeze(0), Z], dim=0)
    for t in range(H): all_pairs.append((Zf[t], A[t], Zf[t+1]))
t0 = time.time()
for ep in range(100):
    idxs = torch.randperm(len(all_pairs))
    losses = []
    for i in range(0, len(idxs), 512):
        idx = idxs[i:i+512]
        zs = torch.stack([all_pairs[j][0] for j in idx]).to(DEVICE)
        ac = torch.stack([all_pairs[j][1] for j in idx]).to(DEVICE)
        nxt = torch.stack([all_pairs[j][2] for j in idx]).to(DEVICE)
        loss = nn.MSELoss()(rl(zs, ac), nxt)
        opt_rl.zero_grad(); loss.backward(); opt_rl.step()
        losses.append(loss.item())
    if (ep+1) % 20 == 0: print(f"  ep {ep+1:3d}: loss={np.mean(losses):.6f}")
print(f"  Done ({time.time()-t0:.1f}s)")

# ── 4. Evaluate drift ──────────────────────────────────────
def solve_slots(z0, A, steps=50, lr=0.01, init_Z=None):
    B, Hf = A.shape[:2]
    Z = (init_Z.clone().detach() if init_Z is not None
         else torch.randn(B, Hf, N_slots, d_slot, device=DEVICE))
    Z.requires_grad_(True); vel = torch.zeros_like(Z)
    for _ in range(steps):
        e = iwcm(z0, A, Z).mean()
        g = torch.autograd.grad(e, Z, create_graph=False)[0]
        vel = 0.9 * vel + g
        Z = Z.detach() - lr * vel
        Z.requires_grad_(True); vel = vel.detach()
    return Z

measure_steps = list(range(10, H+1, 10))
results = {t: [] for t in measure_steps}
n_test = min(len(test_valid), 80)

print(f"Evaluating {n_test} test trajectories...")
t0 = time.time()
for idx in range(n_test):
    if (idx+1) % 20 == 0: print(f"  {idx+1}/{n_test}...")
    z0, A, Z_true = test_valid[idx]
    z0_b, A_b, Z_true_b = z0.unsqueeze(0).to(DEVICE), A.unsqueeze(0).to(DEVICE), Z_true.unsqueeze(0).to(DEVICE)
    with torch.no_grad(): Z_rollout = rl.rollout(z0_b, A_b)
    Z_warm = solve_slots(z0_b, A_b, init_Z=Z_rollout, steps=100, lr=0.005)
    for t in measure_steps:
        with torch.no_grad():
            e_true = iwcm(z0_b, A_b[:, :t], Z_true_b[:, :t]).item()
            e_roll = iwcm(z0_b, A_b[:, :t], Z_rollout[:, :t]).item()
            e_warm = iwcm(z0_b, A_b[:, :t], Z_warm[:, :t]).item()
        results[t].append((e_true, e_roll, e_warm))
print(f"  Done ({time.time()-t0:.1f}s)")

# ── 5. Print + CSV ─────────────────────────────────────────
print("\n" + "=" * 75)
print(f"{'Step':>6} | {'E(true)':>9} {'E(roll)':>9} {'E(warm)':>9} | {'Δ roll':>9} {'Δ warm':>9}")
print("-" * 75)
for t in measure_steps:
    v = results[t]; n_ok = len(v)
    e_true = np.mean([x[0] for x in v]) if n_ok else 0
    e_roll = np.mean([x[1] for x in v]) if n_ok else 0
    e_warm = np.mean([x[2] for x in v]) if n_ok else 0
    print(f"  {t:4d} | {e_true:9.3f} {e_roll:9.3f} {e_warm:9.3f} | "
          f"{e_roll-e_true:+9.3f} {e_warm-e_true:+9.3f}")

print("\nCSV:")
print("step,true_energy,rollout_energy,warm_energy,rollout_drift,warm_drift")
for t in measure_steps:
    v = results[t]; n_ok = len(v)
    if n_ok:
        e_true = np.mean([x[0] for x in v])
        e_roll = np.mean([x[1] for x in v])
        e_warm = np.mean([x[2] for x in v])
        print(f"{t},{e_true:.4f},{e_roll:.4f},{e_warm:.4f},{e_roll-e_true:+.4f},{e_warm-e_true:+.4f}")

print("\n" + "=" * 75)
t100 = results[100] if 100 in results else results[max(results.keys())]
dr = np.mean([x[1]-x[0] for x in t100])
dw = np.mean([x[2]-x[0] for x in t100])
print(f"At H=100: rollout Δ={dr:+.3f}, warm-start IWCM Δ={dw:+.3f}")
print("=" * 75)
