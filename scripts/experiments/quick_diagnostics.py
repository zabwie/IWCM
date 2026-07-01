#!/usr/bin/env python3
"""Quick targeted diagnostics for cheetah/walker IWCM convergence."""
import sys, torch, numpy as np
from pathlib import Path
BASE = str(Path(__file__).resolve().parent.parent.parent)
sys.path.insert(0, BASE)
torch.set_num_threads(1)
from scripts.experiments._common import *
import torch.nn.functional as F

DEV = torch.device('cuda')
H = 100

def train_iwcm_long(tv, tc, d_slot=19, d_a=1, Ns=8, epochs=500, domain=""):
    """Train longer and track ev/ec every epoch."""
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
        if (ep+1) % 50 == 0 or ep == 0:
            ev_m = ev.mean().item(); ec_m = ec.mean().item()
            print(f"  {domain}: ep {ep+1:4d}: ev={ev_m:+.3f} ec={ec_m:+.3f} gap={ec_m-ev_m:+.3f}")
    m.eval(); return m

# ── Cheetah: longer training ──
print("="*60)
print("CHEETAH: 500 epochs")
print("="*60)
torch.manual_seed(42); np.random.seed(42)
w, e, da = env('cheetah', 'run')
tv, tc, ts = data(w, e, nv=400, nc=200)
print(f"  Data: {len(tv)} valid, {len(tc)} corrupt, {len(ts)} test")
m = train_iwcm_long(tv, tc, d_a=da, epochs=500, domain="cheetah")

# Evaluate
print(f"\n  ── Cheetah final evaluation ──")
z0, A, Zt = ts[0]
zb = z0.unsqueeze(0).to(DEV); Ab = A.unsqueeze(0).to(DEV); Zb = Zt.unsqueeze(0).to(DEV)
Z_init = zb.unsqueeze(1).expand(-1, H, -1, -1).clone().to(DEV)
Z_k100 = solve(m, zb, Ab, 100, 0.01, Z_init)
Z_k20 = solve(m, zb, Ab, 20, 0.08, Z_init)
with torch.no_grad():
    print(f"  GT energy:    {m(zb, Ab, Zb).item():+.3f}")
    print(f"  K100 energy:  {m(zb, Ab, Z_k100).item():+.3f}")
    print(f"  K20 energy:   {m(zb, Ab, Z_k20).item():+.3f}")

# ── Walker: check data quality ──
print("\n" + "="*60)
print("WALKER: data quality check")
print("="*60)
torch.manual_seed(42); np.random.seed(42)
w, e, da = env('walker', 'walk')
tv, tc, ts = data(w, e, nv=400, nc=200)
print(f"  Data: {len(tv)} valid, {len(tc)} corrupt, {len(ts)} test")

# Check raw trajectory stats
for name, data_list in [('valid', tv), ('corrupt', tc)]:
    energies = []
    for z0, A, Zt in data_list[:20]:
        with torch.no_grad():
            zb = z0.unsqueeze(0).to(DEV); Ab = A.unsqueeze(0).to(DEV); Zb = Zt.unsqueeze(0).to(DEV)
            Z_init = zb.unsqueeze(1).expand(-1, H, -1, -1).clone().to(DEV)
            Z_zero = torch.zeros_like(Z_init)
        # Quick pre-check: does the data have variance?
        z_norm = Zt.norm().item()
        energies.append(z_norm)
    print(f"  {name}: trajectory norm μ={np.mean(energies):.4f} σ={np.std(energies):.4f}")

# Check corruption types used
for ct in ['teleport', 'freeze', 'reverse']:
    for _ in range(5):
        r = w.generate_corrupted_trajectory(H, corruption_type=ct, rng=np.random.RandomState(42))
        if r is not None:
            ps, _, A = r[0], r[1], r[2]
            qpos0, qvel0 = ps[0]
            qposH, qvelH = ps[min(H, len(ps))-1]
            delta_qpos = np.max(np.abs(qposH - qpos0))
            print(f"  {ct}: qpos drift from step 0 to {H}: {delta_qpos:.4f}")
            break

# Train walker longer
print(f"\n  ── Walker: 500 epochs ──")
torch.manual_seed(42); np.random.seed(42)
m_walker = train_iwcm_long(tv, tc, d_a=da, epochs=500, domain="walker")

# Evaluate
z0, A, Zt = ts[0]
zb = z0.unsqueeze(0).to(DEV); Ab = A.unsqueeze(0).to(DEV); Zb = Zt.unsqueeze(0).to(DEV)
Z_init = zb.unsqueeze(1).expand(-1, H, -1, -1).clone().to(DEV)
Z_k100 = solve(m_walker, zb, Ab, 100, 0.01, Z_init)
Z_k20 = solve(m_walker, zb, Ab, 20, 0.08, Z_init)
with torch.no_grad():
    print(f"  GT energy:    {m_walker(zb, Ab, Zb).item():+.3f}")
    print(f"  K100 energy:  {m_walker(zb, Ab, Z_k100).item():+.3f}")
    print(f"  K20 energy:   {m_walker(zb, Ab, Z_k20).item():+.3f}")
