#!/usr/bin/env python3
"""Multi-start solver: test cold-start recovery via cheap probes + commitment.
   ponytail: one-shot, hardcoded, minimum code."""
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

print(f"Multi-start solver test (H={H}, {DEVICE})")
set_seed(42); rng = np.random.RandomState(42)

# ── 1. Data + IWCM (same pipeline as before) ────────────────
wrapper = DMControlWrapper('cartpole', 'swingup', seed=42, max_episode_steps=300)
encoder = DMControlOracleEncoder('cartpole'); da = wrapper.action_dim

def gen(horizon=H, corrupt=None):
    r = (wrapper.generate_corrupted_trajectory(horizon, corruption_type=corrupt, rng=rng)
         if corrupt else wrapper.generate_trajectory(horizon, random_policy=True))
    if r is None: return None
    enc = encoder.encode_trajectory(r[0], r[2], horizon)
    if enc is None: return None
    return (torch.from_numpy(enc[0]).float(), torch.from_numpy(enc[1]).float(),
            torch.from_numpy(enc[2]).float())

print("Data + IWCM training...")
t0 = time.time()
tv = [gen() for _ in range(400)]; tv = [t for t in tv if t is not None]
tc = [gen(corrupt=ct) for ct in ['teleport','freeze','reverse'] for _ in range(70)]
tc = [t for t in tc if t is not None][:200]
test_v = [gen() for _ in range(60)]; test_v = [t for t in test_v if t is not None]
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
print(f"  {time.time()-t0:.1f}s")

# ── 2. Solvers ──────────────────────────────────────────────
def solve_cold(z0, A, steps=100, lr=0.01):
    """Standard cold-start GD."""
    B, Hf = A.shape[:2]
    Z = torch.randn(B, Hf, Ns, ds, device=DEVICE, requires_grad=True)
    vel = torch.zeros_like(Z)
    for _ in range(steps):
        e = iwcm(z0, A, Z).mean()
        g = torch.autograd.grad(e, Z, create_graph=False)[0]
        vel = 0.9 * vel + g
        Z = Z.detach() - lr * vel
        Z.requires_grad_(True); vel = vel.detach()
    return Z

def solve_multistart(z0, A, N=20, probe_K=3, full_K=100, lr=0.01):
    """Multi-start with commitment: try N cheap probes, commit to best."""
    B, Hf = A.shape[:2]; best_Z = None; best_e = float('inf')
    for i in range(N):
        Z = torch.randn(B, Hf, Ns, ds, device=DEVICE, requires_grad=True)
        vel = torch.zeros_like(Z)
        # Cheap probe — just K_short steps
        for _ in range(probe_K):
            e = iwcm(z0, A, Z).mean()
            g = torch.autograd.grad(e, Z, create_graph=False)[0]
            vel = 0.9 * vel + g
            Z = Z.detach() - lr * vel
            Z.requires_grad_(True); vel = vel.detach()
        with torch.no_grad():
            e_probe = iwcm(z0, A, Z).mean().item()
        if e_probe < best_e:
            best_e = e_probe; best_Z = Z.clone().detach()
    # Commit: full refinement from best candidate
    Z = best_Z.requires_grad_(True); vel = torch.zeros_like(Z)
    for _ in range(full_K):
        e = iwcm(z0, A, Z).mean()
        g = torch.autograd.grad(e, Z, create_graph=False)[0]
        vel = 0.9 * vel + g
        Z = Z.detach() - lr * vel
        Z.requires_grad_(True); vel = vel.detach()
    return Z

def solve_warm(z0, A, init_Z, steps=100, lr=0.005):
    """Warm-start from a given initialization."""
    B, Hf = A.shape[:2]
    Z = init_Z.clone().detach().requires_grad_(True)
    vel = torch.zeros_like(Z)
    for _ in range(steps):
        e = iwcm(z0, A, Z).mean()
        g = torch.autograd.grad(e, Z, create_graph=False)[0]
        vel = 0.9 * vel + g
        Z = Z.detach() - lr * vel
        Z.requires_grad_(True); vel = vel.detach()
    return Z

# ── 3. Compare ──────────────────────────────────────────────
# Also train a quick rollout model for warm-start baseline
class QuickRollout(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(Ns*ds+da,128), nn.ReLU(), nn.Linear(128, Ns*ds))
    def forward(self,z,a): B=z.shape[0]; return self.net(torch.cat([z.reshape(B,-1),a],dim=-1)).reshape(B,Ns,ds)
    def rollout(self,z0,A):
        B,Hf=A.shape[:2]; z=z0; zs=[]
        for t in range(Hf): z=self(z,A[:,t]); zs.append(z)
        return torch.stack(zs,dim=1)
rl = QuickRollout().to(DEVICE); o = torch.optim.Adam(rl.parameters(),lr=1e-3)
pairs = [(torch.cat([zt.unsqueeze(0),Z],dim=0)[t], A[t], torch.cat([zt.unsqueeze(0),Z],dim=0)[t+1])
         for zt,A,Z in tv for t in range(H)]
for ep in range(50):
    idxs=torch.randperm(len(pairs)); ls=[]
    for i in range(0,len(idxs),512):
        idx=idxs[i:i+512]
        zs=torch.stack([pairs[j][0] for j in idx]).to(DEVICE)
        ac=torch.stack([pairs[j][1] for j in idx]).to(DEVICE)
        nx=torch.stack([pairs[j][2] for j in idx]).to(DEVICE)
        l=nn.MSELoss()(rl(zs,ac),nx); o.zero_grad(); l.backward(); o.step(); ls.append(l.item())

measure = list(range(10, H+1, 10))
results = {t: {'cold':[], 'multi':[], 'warm':[], 'true':[]} for t in measure}

print(f"Evaluating {len(test_v)} trajectories...")
t0 = time.time()
for idx in range(len(test_v)):
    if (idx+1) % 20 == 0: print(f"  {idx+1}/{len(test_v)}...")
    z0, A, Z_true = test_v[idx]
    zb, Ab, Zb = z0.unsqueeze(0).to(DEVICE), A.unsqueeze(0).to(DEVICE), Z_true.unsqueeze(0).to(DEVICE)
    Z_cold = solve_cold(zb, Ab)
    Z_multi = solve_multistart(zb, Ab)
    with torch.no_grad(): Z_roll = rl.rollout(zb, Ab)
    Z_warm = solve_warm(zb, Ab, Z_roll)
    for t in measure:
        with torch.no_grad():
            results[t]['true'].append(iwcm(zb, Ab[:,:t], Zb[:,:t]).item())
            results[t]['cold'].append(iwcm(zb, Ab[:,:t], Z_cold[:,:t]).item())
            results[t]['multi'].append(iwcm(zb, Ab[:,:t], Z_multi[:,:t]).item())
            results[t]['warm'].append(iwcm(zb, Ab[:,:t], Z_warm[:,:t]).item())
print(f"  {time.time()-t0:.1f}s")

# ── 4. Results ──────────────────────────────────────────────
print("\n" + "=" * 100)
print(f"{'Step':>6} | {'E(true)':>9} {'E(cold)':>9} {'E(multi)':>9} {'E(warm)':>9} | "
      f"{'Δ cold':>9} {'Δ multi':>9} {'Δ warm':>9}")
print("-" * 100)
for t in measure:
    v = results[t]
    et = np.mean(v['true']); ec = np.mean(v['cold']); em = np.mean(v['multi']); ew = np.mean(v['warm'])
    print(f"  {t:4d} | {et:9.3f} {ec:9.3f} {em:9.3f} {ew:9.3f} | "
          f"{ec-et:+9.3f} {em-et:+9.3f} {ew-et:+9.3f}")

# Summary
print("-" * 100)
v = results[100]
ec = np.mean(v['cold']); em = np.mean(v['multi']); ew = np.mean(v['warm']); et = np.mean(v['true'])
print(f"H=100: cold Δ={ec-et:+.3f}, multi-start Δ={em-et:+.3f}, warm Δ={ew-et:+.3f}")
print(f"Multi-start eliminates {(1-(em-et)/(ec-et))*100:.0f}% of cold-start failure" if ec!=et else "")
print(f"vs warm-start: multi-start Δ={em-et:+.3f} vs warm Δ={ew-et:+.3f} ({'better' if em<ew else 'worse'})")
print("=" * 100)

print("\nCSV:")
print("step,true_energy,cold_energy,multi_energy,warm_energy,cold_drift,multi_drift,warm_drift")
for t in measure:
    v = results[t]
    et=np.mean(v['true']); ec=np.mean(v['cold']); em=np.mean(v['multi']); ew=np.mean(v['warm'])
    print(f"{t},{et:.4f},{ec:.4f},{em:.4f},{ew:.4f},{ec-et:+.4f},{em-et:+.4f},{ew-et:+.4f}")
