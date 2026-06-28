#!/usr/bin/env python3
"""Recovery test: deliberately degraded rollout → IWCM warm-start recovers?
   ponytail: one-shot, hardcoded paths, minimum code."""
import sys, time, torch, numpy as np
from pathlib import Path
BASE = str(Path(__file__).resolve().parent.parent)
sys.path.insert(0, BASE)
torch.set_num_threads(1)
import torch.nn as nn, torch.nn.functional as F
from src.env.dm_control_wrapper import DMControlWrapper
from src.env.dm_control_encoder import DMControlOracleEncoder, MAX_BODIES, ORACLE_SLOT_DIM
from src.iwcm.fused_energy import FusedIWCMEnergy
from src.utils.seed import set_seed
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
H = 100; Ns = MAX_BODIES; ds = ORACLE_SLOT_DIM

print(f"Recovery experiment (H={H}, {DEVICE})")
set_seed(42); rng = np.random.RandomState(42)

# ── 1. Generate data & train a GOOD IWCM (same as before) ──
wrapper = DMControlWrapper('cartpole', 'swingup', seed=42, max_episode_steps=300)
encoder = DMControlOracleEncoder('cartpole')
da = wrapper.action_dim

def gen(horizon=H, corrupt=None):
    r = (wrapper.generate_corrupted_trajectory(horizon, corruption_type=corrupt, rng=rng)
         if corrupt else wrapper.generate_trajectory(horizon, random_policy=True))
    if r is None: return None
    ps = r[0]; ac = r[2] if not corrupt else r[2]
    enc = encoder.encode_trajectory(ps, ac, horizon)
    if enc is None: return None
    return (torch.from_numpy(enc[0]).float(), torch.from_numpy(enc[1]).float(),
            torch.from_numpy(enc[2]).float())

print("Generating data + training IWCM...")
t0 = time.time()
tv = [gen() for _ in range(400)]; tv = [t for t in tv if t is not None]
tc = [gen(corrupt=ct) for ct in ['teleport','freeze','reverse'] for _ in range(70)]
tc = [t for t in tc if t is not None][:200]
test_v = [gen() for _ in range(80)]; test_v = [t for t in test_v if t is not None]
iwcm = FusedIWCMEnergy(d_slot=ds, d_action=da, hidden=128, num_slots=Ns).to(DEVICE)
opt = torch.optim.Adam(iwcm.parameters(), lr=1e-3)
for ep in range(150):
    vi = np.random.choice(len(tv), 64, replace=False)
    ci = np.random.choice(len(tc), 64, replace=False)
    vz0 = torch.stack([tv[i][0] for i in vi]).to(DEVICE)
    vA = torch.stack([tv[i][1] for i in vi]).to(DEVICE)
    vZ = torch.stack([tv[i][2] for i in vi]).to(DEVICE)
    cz0 = torch.stack([tc[i][0] for i in ci]).to(DEVICE)
    cA = torch.stack([tc[i][1] for i in ci]).to(DEVICE)
    cZ = torch.stack([tc[i][2] for i in ci]).to(DEVICE)
    ev = iwcm(vz0, vA, vZ); ec = iwcm(cz0, cA, cZ)
    loss = F.relu(ev+1).mean() + F.relu(1-ec).mean() + 0.001*(ev.pow(2).mean()+ec.pow(2).mean())
    opt.zero_grad(); loss.backward(); opt.step()
iwcm.eval()
print(f"  IWCM ready ({time.time()-t0:.1f}s)")

# ── 2. Train DEGRADED rollout model ─────────────────────────
# ponytail: deliberately crippled — tiny net, little data, few epochs
print("Training DEGRADED rollout (1×32 hidden, 50 trajs, 15 epochs)...")
class DegradedRollout(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(Ns*ds + da, 32), nn.Tanh(),
            nn.Linear(32, Ns*ds),
        )
    def forward(self, z, a):
        B = z.shape[0]
        return self.net(torch.cat([z.reshape(B,-1), a], dim=-1)).reshape(B, Ns, ds)
    def rollout(self, z0, A):
        B, Hf = A.shape[:2]; z = z0; zs = []
        for t in range(Hf): z = self(z, A[:,t]); zs.append(z)
        return torch.stack(zs, dim=1)

rl = DegradedRollout().to(DEVICE)
opt_rl = torch.optim.Adam(rl.parameters(), lr=1e-3)
# Use only 50 trajectories for training
train_subset = tv[:50]
pairs = []
for z0, A, Z in train_subset:
    Zf = torch.cat([z0.unsqueeze(0), Z], dim=0)
    for t in range(H): pairs.append((Zf[t], A[t], Zf[t+1]))
for ep in range(15):
    idxs = torch.randperm(len(pairs)); losses = []
    for i in range(0, len(idxs), 256):
        idx = idxs[i:i+256]
        zs = torch.stack([pairs[j][0] for j in idx]).to(DEVICE)
        ac = torch.stack([pairs[j][1] for j in idx]).to(DEVICE)
        nx = torch.stack([pairs[j][2] for j in idx]).to(DEVICE)
        loss = nn.MSELoss()(rl(zs,ac), nx)
        opt_rl.zero_grad(); loss.backward(); opt_rl.step()
        losses.append(loss.item())
    if (ep+1) % 5 == 0: print(f"  ep {ep+1:2d}: loss={np.mean(losses):.6f}")
print("  Degraded rollout ready")

# ── 3. Test recovery ────────────────────────────────────────
def solve(z0, A, steps=100, lr=0.005, init_Z=None):
    B, Hf = A.shape[:2]
    Z = (init_Z.clone().detach() if init_Z is not None
         else torch.randn(B, Hf, Ns, ds, device=DEVICE))
    Z.requires_grad_(True); vel = torch.zeros_like(Z)
    for _ in range(steps):
        with torch.enable_grad():
            e = iwcm(z0, A, Z).mean()
        g = torch.autograd.grad(e, Z, create_graph=False)[0]
        vel = 0.9 * vel + g
        Z = Z.detach() - lr * vel
        Z.requires_grad_(True); vel = vel.detach()
    return Z

steps = list(range(10, H+1, 10))
results = {t: [] for t in steps}

# Also track at which step the degraded rollout first exceeds a drift threshold
print("Evaluating recovery...")
t0 = time.time()
for idx in range(min(len(test_v), 50)):
    z0, A, Z_true = test_v[idx]
    z0_b, A_b, Zt_b = z0.unsqueeze(0).to(DEVICE), A.unsqueeze(0).to(DEVICE), Z_true.unsqueeze(0).to(DEVICE)
    with torch.no_grad(): Z_roll = rl.rollout(z0_b, A_b)
    Z_warm = solve(z0_b, A_b, init_Z=Z_roll)
    for t in steps:
        with torch.no_grad():
            et = iwcm(z0_b, A_b[:,:t], Zt_b[:,:t]).item()
            er = iwcm(z0_b, A_b[:,:t], Z_roll[:,:t]).item()
            ew = iwcm(z0_b, A_b[:,:t], Z_warm[:,:t]).item()
        results[t].append((et, er, ew))
print(f"  Done ({time.time()-t0:.1f}s)")

# ── 4. Results ──────────────────────────────────────────────
print("\n" + "=" * 85)
print(f"{'Step':>6} | {'E(true)':>9} {'E(roll)':>9} {'E(warm)':>9} | {'Δ roll':>9} {'Δ warm':>9} | {'Recovered?':>10}")
print("-" * 85)
recovered_steps = []
for t in steps:
    v = results[t]
    et = np.mean([x[0] for x in v]); er = np.mean([x[1] for x in v]); ew = np.mean([x[2] for x in v])
    dr = er - et; dw = ew - et
    recovered = dw < dr  # IWCM closer to true than rollout
    if recovered: recovered_steps.append(t)
    print(f"  {t:4d} | {et:9.3f} {er:9.3f} {ew:9.3f} | {dr:+9.3f} {dw:+9.3f} | {'✓' if recovered else '✗':>10}")

print("-" * 85)
v100 = results[100]
dr100 = np.mean([x[1]-x[0] for x in v100])
dw100 = np.mean([x[2]-x[0] for x in v100])
print(f"H=100: rollout Δ={dr100:+.3f}, warm-start Δ={dw100:+.3f}")
print(f"Recovery: {'YES — IWCM recovers from drifted rollout' if dw100 < dr100 else 'PARTIAL — IWCM improves but does not fully recover'}")
print(f"Divergence gap ratio (rollout/IWCM): {dr100/dw100:.1f}x" if dw100 != 0 else "N/A")
print("=" * 85)

# CSV
print("\nCSV:")
print("step,true_energy,rollout_energy,warm_energy,rollout_drift,warm_drift,recovered")
for t in steps:
    v = results[t]
    et = np.mean([x[0] for x in v]); er = np.mean([x[1] for x in v]); ew = np.mean([x[2] for x in v])
    print(f"{t},{et:.4f},{er:.4f},{ew:.4f},{er-et:+.4f},{ew-et:+.4f},{'1' if ew-et < er-et else '0'}")
