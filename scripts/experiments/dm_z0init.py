#!/usr/bin/env python3
"""Compare init strategies: z0-replication vs random vs warm-start for IWCM solver.
   ponytail: one-shot, hardcoded paths."""
import sys, torch, numpy as np
from pathlib import Path
BASE = str(Path(__file__).resolve().parent.parent)
sys.path.insert(0, BASE)
torch.set_num_threads(1)
import torch.nn as nn, torch.nn.functional as F
from src.env.dm_control_wrapper import DMControlWrapper
from src.env.dm_control_encoder import DMControlOracleEncoder, MAX_BODIES, ORACLE_SLOT_DIM
from src.iwcm.fused_energy import FusedIWCMEnergy
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
H = 100; Ns = MAX_BODIES; ds = ORACLE_SLOT_DIM

# ── 1. Data + IWCM ──────────────────────────────────────────
wrapper = DMControlWrapper('cartpole','swingup',seed=42,max_episode_steps=300)
encoder = DMControlOracleEncoder('cartpole'); da = wrapper.action_dim

def gen(corrupt=None):
    rng=np.random.RandomState(42)
    r = (wrapper.generate_corrupted_trajectory(H,corruption_type=corrupt,rng=rng)
         if corrupt else wrapper.generate_trajectory(H,random_policy=True))
    if r is None: return None
    enc = encoder.encode_trajectory(r[0],r[2],H)
    return None if enc is None else (torch.from_numpy(enc[0]).float(),torch.from_numpy(enc[1]).float(),torch.from_numpy(enc[2]).float())

tv = [gen() for _ in range(400)]; tv = [t for t in tv if t is not None]
tc = [gen(corrupt=ct) for ct in ['teleport','freeze','reverse'] for _ in range(70)]
tc = [t for t in tc if t is not None][:200]
test_v = [gen() for _ in range(40)]; test_v = [t for t in test_v if t is not None]

iwcm = FusedIWCMEnergy(d_slot=ds,d_action=da,hidden=128,num_slots=Ns).to(DEVICE)
opt = torch.optim.Adam(iwcm.parameters(),lr=1e-3)
for ep in range(150):
    vi=np.random.choice(len(tv),64,replace=False); ci=np.random.choice(len(tc),64,replace=False)
    vz0=torch.stack([tv[i][0] for i in vi]).to(DEVICE); vA=torch.stack([tv[i][1] for i in vi]).to(DEVICE)
    vZ=torch.stack([tv[i][2] for i in vi]).to(DEVICE); cz0=torch.stack([tc[i][0] for i in ci]).to(DEVICE)
    cA=torch.stack([tc[i][1] for i in ci]).to(DEVICE); cZ=torch.stack([tc[i][2] for i in ci]).to(DEVICE)
    ev=iwcm(vz0,vA,vZ); ec=iwcm(cz0,cA,cZ)
    loss=F.relu(ev+1).mean()+F.relu(1-ec).mean()+0.001*(ev.pow(2).mean()+ec.pow(2).mean())
    opt.zero_grad();loss.backward();opt.step()
iwcm.eval()

# Quick rollout for warm-start baseline
class Roll(nn.Module):
    def __init__(self):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(Ns*ds+da,128),nn.ReLU(),nn.Linear(128,Ns*ds))
    def forward(self,z,a): B=z.shape[0]; return self.net(torch.cat([z.reshape(B,-1),a],dim=-1)).reshape(B,Ns,ds)
    def rollout(self,z0,A): B,Hf=A.shape[:2];z=z0;zs=[]; [zs.append(z:=self(z,A[:,t])) for t in range(Hf)]; return torch.stack(zs,dim=1)
rl=Roll().to(DEVICE); o=torch.optim.Adam(rl.parameters(),lr=1e-3)
pairs=[(torch.cat([z.unsqueeze(0),Z],0)[t],A[t],torch.cat([z.unsqueeze(0),Z],0)[t+1]) for z,A,Z in tv for t in range(H)]
for ep in range(30):
    idxs=torch.randperm(len(pairs))
    for i in range(0,len(idxs),512):
        idx=idxs[i:i+512]; zs=torch.stack([pairs[j][0] for j in idx]).to(DEVICE)
        ac=torch.stack([pairs[j][1] for j in idx]).to(DEVICE); nx=torch.stack([pairs[j][2] for j in idx]).to(DEVICE)
        l=nn.MSELoss()(rl(zs,ac),nx); o.zero_grad();l.backward();o.step()

# ── 2. Solvers ──────────────────────────────────────────────
def solve(z0, A, init_type='random', steps=100, lr=0.01):
    B, Hf = A.shape[:2]
    if init_type == 'random':
        Z = torch.randn(B, Hf, Ns, ds, device=DEVICE)
    elif init_type == 'z0rep':
        Z = z0.unsqueeze(1).expand(-1, Hf, -1, -1).clone()
    elif init_type == 'warm':
        return Z_warm  # set externally
    Z = Z.detach().requires_grad_(True); vel = torch.zeros_like(Z)
    for _ in range(steps):
        e = iwcm(z0, A, Z).mean()
        g = torch.autograd.grad(e, Z, create_graph=False)[0]
        vel = 0.9 * vel + g
        Z = Z.detach() - lr * vel
        Z.requires_grad_(True); vel = vel.detach()
    return Z

# ── 3. Evaluate ─────────────────────────────────────────────
steps_m = list(range(10, H+1, 10))
results = {t: {'true':[],'cold':[],'z0rep':[],'warm':[]} for t in steps_m}
n_test = len(test_v)

for idx in range(n_test):
    z0, A, Zt = test_v[idx]
    zb, Ab, Ztb = z0.unsqueeze(0).to(DEVICE), A.unsqueeze(0).to(DEVICE), Zt.unsqueeze(0).to(DEVICE)
    Z_cold = solve(zb, Ab, 'random')
    Z_z0 = solve(zb, Ab, 'z0rep')
    with torch.no_grad(): Z_rl = rl.rollout(zb, Ab)
    # Need to make solve work for warm-start — just run inline
    Z = Z_rl.clone().detach().requires_grad_(True); vel = torch.zeros_like(Z)
    for _ in range(100):
        e = iwcm(zb, Ab, Z).mean()
        g = torch.autograd.grad(e, Z, create_graph=False)[0]
        vel = 0.9 * vel + g
        Z = Z.detach() - 0.005 * vel
        Z.requires_grad_(True); vel = vel.detach()
    Z_warm = Z
    for t in steps_m:
        with torch.no_grad():
            results[t]['true'].append(iwcm(zb, Ab[:,:t], Ztb[:,:t]).item())
            results[t]['cold'].append(iwcm(zb, Ab[:,:t], Z_cold[:,:t]).item())
            results[t]['z0rep'].append(iwcm(zb, Ab[:,:t], Z_z0[:,:t]).item())
            results[t]['warm'].append(iwcm(zb, Ab[:,:t], Z_warm[:,:t]).item())

# ── 4. Results ──────────────────────────────────────────────
print("\n" + "=" * 110)
hdr = f"{'Step':>6} | {'E(true)':>9} {'E(cold)':>9} {'E(z0rep)':>9} {'E(warm)':>9} | {'Δ cold':>9} {'Δ z0rep':>9} {'Δ warm':>9}"
print(hdr); print("-" * 110)
for t in steps_m:
    v = results[t]
    et=np.mean(v['true']); ec=np.mean(v['cold']); ez=np.mean(v['z0rep']); ew=np.mean(v['warm'])
    print(f"  {t:4d} | {et:9.3f} {ec:9.3f} {ez:9.3f} {ew:9.3f} | "
          f"{ec-et:+9.3f} {ez-et:+9.3f} {ew-et:+9.3f}")

print("-" * 110)
v = results[100]
et=np.mean(v['true']); ec=np.mean(v['cold']); ez=np.mean(v['z0rep']); ew=np.mean(v['warm'])
print(f"H=100: cold Δ={ec-et:+.3f}, z0rep Δ={ez-et:+.3f}, warm Δ={ew-et:+.3f}")
print(f"z0rep eliminates {(1-(ez-et)/(ec-et))*100:.0f}% of cold-start drift" if ec!=et else "")
print(f"z0rep vs warm: {'z0rep better' if ez<ew else 'warm better'} (Δ={ez-et:+.3f} vs {ew-et:+.3f})")

print("\nCSV:")
print("step,true_energy,cold_energy,z0rep_energy,warm_energy,cold_drift,z0rep_drift,warm_drift")
for t in steps_m:
    v=results[t]; et=np.mean(v['true']); ec=np.mean(v['cold']); ez=np.mean(v['z0rep']); ew=np.mean(v['warm'])
    print(f"{t},{et:.4f},{ec:.4f},{ez:.4f},{ew:.4f},{ec-et:+.4f},{ez-et:+.4f},{ew-et:+.4f}")
