"""Shared helpers for drift experiments. ponytail: delete if not reused."""
import sys, torch, numpy as np
from pathlib import Path
BASE = str(Path(__file__).resolve().parent.parent.parent)
sys.path.insert(0, BASE)
import torch.nn as nn, torch.nn.functional as F
from src.env.dm_control_wrapper import DMControlWrapper
from src.env.dm_control_encoder import DMControlOracleEncoder
from src.iwcm.fused_energy import FusedIWCMEnergy

DEV = 'cuda' if torch.cuda.is_available() else 'cpu'

def env(name='cartpole', task='swingup', seed=42):
    w = DMControlWrapper(name, task, seed=seed, max_episode_steps=300)
    e = DMControlOracleEncoder(name)
    return w, e, w.action_dim

def gen(w, e, H=100, corrupt=None, rng=None):
    r = (w.generate_corrupted_trajectory(H, corruption_type=corrupt, rng=rng or np.random.RandomState(42))
         if corrupt else w.generate_trajectory(H, random_policy=True))
    if r is None: return None
    enc = e.encode_trajectory(r[0], r[2], H)
    return None if enc is None else tuple(torch.from_numpy(x).float() for x in enc)

def data(w, e, nv=400, nc=200, H=100):
    rng = np.random.RandomState(42)
    tv = [gen(w, e, H) for _ in range(nv)]; tv = [t for t in tv if t is not None]
    # ponytail: teleport skipped for fast-moving domains where slot signal is undetectable
    corrupts = ['freeze', 'reverse'] if w.domain_name == 'cheetah' else ['teleport', 'freeze', 'reverse']
    tc = [gen(w, e, H, corrupt=ct, rng=rng) for ct in corrupts for _ in range(999)]
    tc = [t for t in tc if t is not None][:nc]
    ts = [gen(w, e, H) for _ in range(999)]; ts = [t for t in ts if t is not None][:80]
    return tv, tc, ts

def train_iwcm(tv, tc, d_slot=19, d_a=1, Ns=8, epochs=150):
    m = FusedIWCMEnergy(d_slot, d_a, hidden=128, num_slots=Ns).to(DEV)
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    for ep in range(epochs):
        vi = np.random.choice(len(tv), 64, replace=False)
        ci = np.random.choice(len(tc), 64, replace=False)
        v = [torch.stack([tv[i][j] for i in vi]).to(DEV) for j in range(3)]
        c = [torch.stack([tc[i][j] for i in ci]).to(DEV) for j in range(3)]
        ev = m(*v); ec = m(*c)
        loss = F.relu(ev+1).mean() + F.relu(1-ec).mean() + 0.001*(ev.pow(2).mean()+ec.pow(2).mean())
        opt.zero_grad(); loss.backward(); opt.step()
        if (ep+1) % 50 == 0: print(f"  ep {ep+1:3d}: ev={ev.mean().item():+.3f} ec={ec.mean().item():+.3f}")
    m.eval(); return m

def solve(m, z0, A, steps=100, lr=0.01, init_Z=None):
    B, Hf = A.shape[:2]; Ns, ds = z0.shape[1], z0.shape[2]
    Z = (init_Z.clone().detach() if init_Z is not None else torch.randn(B, Hf, Ns, ds, device=z0.device))
    Z.requires_grad_(True); vel = torch.zeros_like(Z)
    for _ in range(steps):
        e = m(z0, A, Z).mean()
        g = torch.autograd.grad(e, Z, create_graph=False)[0]
        vel = 0.9 * vel + g
        Z = Z.detach() - lr * vel
        Z.requires_grad_(True); vel = vel.detach()
    return Z

class Rollout(nn.Module):
    def __init__(self, Ns, ds, d_a, h=256):
        super().__init__(); self.net = nn.Sequential(nn.Linear(Ns*ds+d_a, h), nn.ReLU(), nn.Linear(h, Ns*ds))
    def forward(self, z, a): B=z.shape[0]; return self.net(torch.cat([z.reshape(B,-1),a],-1)).reshape(B,*z.shape[1:])
    def rollout(self, z0, A): B,H=A.shape[:2];z=z0;zs=[];[zs.append(z:=self(z,A[:,t])) for t in range(H)]; return torch.stack(zs,1)

def train_rl(rl, tv, H, ep=100):
    opt = torch.optim.Adam(rl.parameters(), lr=1e-3); DEV = next(rl.parameters()).device
    pairs = [(torch.cat([z.unsqueeze(0),Z],0)[t],A[t],torch.cat([z.unsqueeze(0),Z],0)[t+1]) for z,A,Z in tv for t in range(H)]
    for e in range(ep):
        idxs = torch.randperm(len(pairs)); ls = []
        for i in range(0, len(idxs), 512):
            idx = idxs[i:i+512]; zs=torch.stack([pairs[j][0] for j in idx]).to(DEV)
            ac=torch.stack([pairs[j][1] for j in idx]).to(DEV); nx=torch.stack([pairs[j][2] for j in idx]).to(DEV)
            l=nn.MSELoss()(rl(zs,ac),nx); opt.zero_grad();l.backward();opt.step();ls.append(l.item())
        if (e+1)%20==0: print(f"  ep {e+1:3d}: loss={np.mean(ls):.6f}")
