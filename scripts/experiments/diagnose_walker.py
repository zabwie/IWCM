#!/usr/bin/env python3
"""Diagnose why walker energy function gradients vanish."""
import sys, torch, numpy as np
from pathlib import Path
BASE = str(Path(__file__).resolve().parent.parent.parent)
sys.path.insert(0, BASE)
torch.set_num_threads(1)
from scripts.experiments._common import *
import torch.nn.functional as F

DEV = torch.device('cuda')
H = 100

torch.manual_seed(42); np.random.seed(42)
w, e, da = env('walker', 'walk')
tv, tc, ts = data(w, e, nv=400, nc=200)

# Check data: do corrupted trajectories actually differ from valid?
print("=== Data quality check ===")
for name, dl in [('valid', tv[:10]), ('corrupt', tc[:10])]:
    norms = []
    for z0, A, Zt in dl:
        # Check slot channel stats per domain
        pos_slots = Zt[:, :, 5:8]   # position channels
        vel_slots = Zt[:, :, 8:11]  # velocity channels
        norms.append(pos_slots.norm().item())
    print(f"  {name}: pos_norm μ={np.mean(norms):.4f}")

# Test model with gradient tracking
m = FusedIWCMEnergy(19, da, hidden=128, num_slots=8).to(DEV)
opt = torch.optim.Adam(m.parameters(), lr=1e-3)

# First batch
vi = np.random.choice(len(tv), 64, replace=False)
ci = np.random.choice(len(tc), 64, replace=False)
v = [torch.stack([tv[i][j] for i in vi]).to(DEV) for j in range(3)]
c = [torch.stack([tc[i][j] for i in ci]).to(DEV) for j in range(3)]

ev = m(*v)
ec = m(*c)

# Check gradient norms
loss = F.relu(ev+1).mean() + F.relu(1-ec).mean() + 0.001*(ev.pow(2).mean()+ec.pow(2).mean())
loss.backward()

total_norm = 0
for name, p in m.named_parameters():
    if p.grad is not None:
        gn = p.grad.norm().item()
        total_norm += gn
        if gn > 0.001:
            print(f"  {name}: grad_norm={gn:.6f}")
print(f"  Total grad norm: {total_norm:.6f}")

# Check head weight distribution
for name, p in m.named_parameters():
    if 'weight' in name:
        print(f"  {name}: weights μ={p.data.mean().item():.6f} σ={p.data.std().item():.6f}")

print(f"\n  Initial: ev={ev.mean().item():+.3f} ec={ec.mean().item():+.3f}")

# Train 20 steps and check grad norms
for step in range(20):
    vi = np.random.choice(len(tv), 64)
    ci = np.random.choice(len(tc), 64)
    v = [torch.stack([tv[i][j] for i in vi]).to(DEV) for j in range(3)]
    c = [torch.stack([tc[i][j] for i in ci]).to(DEV) for j in range(3)]
    opt.zero_grad()
    ev, ec = m(*v), m(*c)
    # Try BCE
    pv = torch.sigmoid(-ev); pc = torch.sigmoid(-ec)
    loss = F.binary_cross_entropy(pv, torch.ones_like(pv)) + F.binary_cross_entropy(pc, torch.zeros_like(pc))
    loss.backward()
    gn = sum(p.grad.norm().item() for p in m.parameters() if p.grad is not None)
    opt.step()
    with torch.no_grad():
        shared_out = m.shared(v[2]).detach()
        shared_norm = shared_out.norm(dim=-1).mean().item()
    if step < 5 or step % 5 == 0:
        print(f"  step {step:2d}: ev={ev.mean().item():+.3f} ec={ec.mean().item():+.3f} "
              f"grad_norm={gn:.6f} shared_out_norm={shared_norm:.6f}")

# Try different approach: remove LayerNorm, see if training works
print("\n=== Without LayerNorm ===")
torch.manual_seed(42)
m2 = FusedIWCMEnergy(19, da, hidden=128, num_slots=8).to(DEV)
# Remove LayerNorm
def forward_original(self, z0, A, Z):
    B, H, N, d = Z.shape
    Zf = self.shared(Z)
    Z_mean = Zf.mean(dim=1)
    Z_max = Zf.amax(dim=1)
    Z_sq = Zf * Zf
    Z_var = F.relu(Z_sq.mean(dim=1) - Z_mean * Z_mean)
    Z_std = torch.sqrt(Z_var + 1e-5)
    Zs = torch.cat([Z_mean, Z_max, Z_std], dim=-1)
    scores = self.head(Zs)
    agg = 0.3 * scores.mean(dim=1) + 0.7 * scores.amax(dim=1)
    return (agg * self.lambdas).sum(dim=-1)

opt2 = torch.optim.Adam(m2.parameters(), lr=1e-3)
for step in range(50):
    vi = np.random.choice(len(tv), 64)
    ci = np.random.choice(len(tc), 64)
    v = [torch.stack([tv[i][j] for i in vi]).to(DEV) for j in range(3)]
    c = [torch.stack([tc[i][j] for i in ci]).to(DEV) for j in range(3)]
    opt2.zero_grad()
    ev, ec = forward_original(m2, *v), forward_original(m2, *c)
    pv = torch.sigmoid(-ev); pc = torch.sigmoid(-ec)
    loss = F.binary_cross_entropy(pv, torch.ones_like(pv)) + F.binary_cross_entropy(pc, torch.zeros_like(pc))
    loss.backward()
    gn = sum(p.grad.norm().item() for p in m2.parameters() if p.grad is not None)
    opt2.step()
    if step < 5 or step % 10 == 0:
        print(f"  step {step:2d}: ev={ev.mean().item():+.3f} ec={ec.mean().item():+.3f} grad_norm={gn:.6f}")

# Try higher LR
print("\n=== LR=1e-2 ===")
torch.manual_seed(42)
m3 = FusedIWCMEnergy(19, da, hidden=128, num_slots=8).to(DEV)
# Use direct head (remove GELU bottleneck)
m3.head = nn.Linear(128 * 3, 3).to(DEV)
opt3 = torch.optim.Adam(m3.parameters(), lr=1e-2)
for step in range(100):
    vi = np.random.choice(len(tv), 64)
    ci = np.random.choice(len(tc), 64)
    v = [torch.stack([tv[i][j] for i in vi]).to(DEV) for j in range(3)]
    c = [torch.stack([tc[i][j] for i in ci]).to(DEV) for j in range(3)]
    opt3.zero_grad()
    ev, ec = m3(*v), m3(*c)
    pv = torch.sigmoid(-ev); pc = torch.sigmoid(-ec)
    loss = F.binary_cross_entropy(pv, torch.ones_like(pv)) + F.binary_cross_entropy(pc, torch.zeros_like(pc))
    loss.backward()
    gn = sum(p.grad.norm().item() for p in m3.parameters() if p.grad is not None)
    torch.nn.utils.clip_grad_norm_(m3.parameters(), 1.0)
    opt3.step()
    if step < 5 or step % 25 == 0:
        with torch.no_grad():
            shared_norm = m3.shared(v[2]).norm(dim=-1).mean().item()
        print(f"  step {step:2d}: ev={ev.mean().item():+.3f} ec={ec.mean().item():+.3f} "
              f"grad_norm={gn:.6f} shared_norm={shared_norm:.6f}")
    
    if step > 10 and abs(ev.mean().item()) < 0.001 and abs(ec.mean().item()) < 0.001:
        print(f"  → Collapsed at step {step}")
        break
