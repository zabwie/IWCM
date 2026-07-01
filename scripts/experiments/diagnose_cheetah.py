#!/usr/bin/env python3
"""Diagnose why cheetah's energy gap is weak (0.70 vs 7+ for others)."""
import sys, torch, numpy as np
from pathlib import Path
BASE = str(Path(__file__).resolve().parent.parent.parent)
sys.path.insert(0, BASE)
torch.set_num_threads(1)
from scripts.experiments._common import *
import torch.nn.functional as F

DEV = torch.device('cuda')
H, Ns, ds = 100, 8, 19

torch.manual_seed(42); np.random.seed(42)
w, e, da = env('cheetah', 'run')
tv, tc, ts = data(w, e, nv=400, nc=200)

# ── Check 1: Per-corruption-type separation ──
print("=== Check 1: Per-corruption-type energy ===")
for ct in ['teleport', 'freeze', 'reverse']:
    ct_trajs = []
    for _ in range(200):
        r = w.generate_corrupted_trajectory(H, corruption_type=ct, rng=np.random.RandomState(42+_))
        if r is not None:
            enc = e.encode_trajectory(r[0], r[2], H)
            if enc is not None:
                ct_trajs.append(tuple(torch.from_numpy(x).float() for x in enc))
    print(f"  {ct}: {len(ct_trajs)} trajectories")

# ── Check 2: Slot feature differences ──
print("\n=== Check 2: Valid vs corrupt slot differences ===")
# Compute mean slot features for valid and each corruption type
valid_slots = torch.stack([tv[i][2] for i in range(min(100, len(tv)))])  # (N, H, Ns, ds)
print(f"  Valid slots: {valid_slots.shape}")
print(f"  Valid mean:  {valid_slots.mean().item():.6f}")
print(f"  Valid std:   {valid_slots.std().item():.6f}")
print(f"  Valid norm:  {valid_slots.norm(dim=-1).mean().item():.4f}")

# Per-channel stats
for ch in range(ds):
    ch_mean = valid_slots[:, :, :, ch].mean().item()
    ch_std = valid_slots[:, :, :, ch].std().item()
    if abs(ch_mean) > 0.01 or ch_std > 0.01:
        print(f"    ch {ch:2d}: μ={ch_mean:+.4f} σ={ch_std:.4f}")

# Compare specific corrupted trajectory
print("\n  ── Individual trajectory comparison ──")
for i in range(3):
    z0_v, A_v, Zt_v = tv[i]
    z0_c, A_c, Zt_c = tc[i]
    with torch.no_grad():
        zdiff = (Zt_v - Zt_c).norm(dim=-1).mean().item()
        zdiff_max = (Zt_v - Zt_c).norm(dim=-1).max().item()
    print(f"  Traj {i}: valid vs corrupt slot diff: mean={zdiff:.4f} max={zdiff_max:.4f}")

# ── Check 3: Train on each corruption type separately ──
print("\n=== Check 3: Train on each corruption type alone ===")
for ct in ['teleport', 'freeze', 'reverse']:
    # Generate data for this corruption type only
    rng = np.random.RandomState(42)
    tc_ct = [gen(w, e, H, corrupt=ct, rng=rng) for _ in range(999)]
    tc_ct = [t for t in tc_ct if t is not None][:200]
    print(f"\n  {ct}: {len(tc_ct)} corrupted trajectories")
    
    m_ct = FusedIWCMEnergy(ds, da, hidden=128, num_slots=Ns).to(DEV)
    opt = torch.optim.Adam(m_ct.parameters(), lr=1e-3)
    for ep in range(200):
        vi = np.random.choice(len(tv), 64)
        ci = np.random.choice(len(tc_ct), 64)
        v = [torch.stack([tv[i][j] for i in vi]).to(DEV) for j in range(3)]
        c = [torch.stack([tc_ct[i][j] for i in ci]).to(DEV) for j in range(3)]
        ev, ec = m_ct(*v), m_ct(*c)
        loss = F.relu(ev + 3).mean() + F.relu(3 - ec).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        if (ep+1) % 100 == 0:
            print(f"    ep {ep+1:3d}: ev={ev.mean().item():+.3f} ec={ec.mean().item():+.3f} gap={ec.mean().item()-ev.mean().item():.3f}")

# ── Check 4: More epochs with better schedule ──
print("\n=== Check 4: Longer training (1000 ep, cosine LR) ===")
m_long = FusedIWCMEnergy(ds, da, hidden=128, num_slots=Ns).to(DEV)
opt = torch.optim.Adam(m_long.parameters(), lr=1e-3)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=1000)
for ep in range(1000):
    vi = np.random.choice(len(tv), 64)
    ci = np.random.choice(len(tc), 64)
    v = [torch.stack([tv[i][j] for i in vi]).to(DEV) for j in range(3)]
    c = [torch.stack([tc[i][j] for i in ci]).to(DEV) for j in range(3)]
    ev, ec = m_long(*v), m_long(*c)
    loss = F.relu(ev + 3).mean() + F.relu(3 - ec).mean()
    opt.zero_grad(); loss.backward(); opt.step()
    sched.step()
    if (ep+1) % 200 == 0 or ep == 0:
        print(f"  ep {ep+1:4d}: ev={ev.mean().item():+.3f} ec={ec.mean().item():+.3f} gap={ec.mean().item()-ev.mean().item():.3f}")

# Final eval
evals, ecvals = [], []
for _ in range(20):
    vi = np.random.choice(len(tv), 64)
    ci = np.random.choice(len(tc), 64)
    v = [torch.stack([tv[i][j] for i in vi]).to(DEV) for j in range(3)]
    c = [torch.stack([tc[i][j] for i in ci]).to(DEV) for j in range(3)]
    with torch.no_grad():
        evals.append(m_long(*v).mean().item())
        ecvals.append(m_long(*c).mean().item())
print(f"  Final: ev={np.mean(evals):+.3f} ec={np.mean(ecvals):+.3f} gap={np.mean(ecvals)-np.mean(evals):.3f}")
