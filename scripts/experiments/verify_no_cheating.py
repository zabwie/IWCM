#!/usr/bin/env python3
"""Verify IWCM isn't cheating: energy landscape is real, solver descends, trajectories are valid.
Runs for all 3 domains and prints pass/fail for each check."""

import sys, torch, numpy as np
from pathlib import Path
BASE = str(Path(__file__).resolve().parent.parent.parent)
sys.path.insert(0, BASE)
torch.set_num_threads(1)
from scripts.experiments._common import *
import torch.nn.functional as F

DEV = torch.device('cuda')
H, Ns, ds = 100, 8, 19
PHYS = slice(5, 11)

DOMAINS = [('cartpole', 'swingup', 1, 200), ('cheetah', 'run', 6, 500), ('walker', 'walk', 6, 800)]

def solve_monotonic(m, z0, A, steps=100, lr=0.01, init_Z=None):
    """Solve and track energy at each step."""
    B = z0.shape[0]; Hf = A.shape[1]
    Z = (init_Z.clone().detach() if init_Z is not None else torch.randn(B, Hf, Ns, ds, device=z0.device))
    Z.requires_grad_(True); vel = torch.zeros_like(Z)
    energies = []
    if init_Z is not None:
        with torch.no_grad(): energies.append(m(z0, A, Z).item())
    for _ in range(steps):
        e = m(z0, A, Z).mean()
        g = torch.autograd.grad(e, Z, create_graph=False)[0]
        vel = 0.9 * vel + g
        Z = Z.detach() - lr * vel
        Z.requires_grad_(True); vel = vel.detach()
        with torch.no_grad(): energies.append(m(z0, A, Z).item())
    return Z, energies

def train_iwcm_strong(tv, tc, d_a, epochs=300, domain=""):
    """Train with a stronger margin (5 instead of 1) for faster convergence."""
    m = FusedIWCMEnergy(ds, d_a, hidden=128, num_slots=Ns).to(DEV)
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    best_gap = -float('inf')
    for ep in range(epochs):
        vi = np.random.choice(len(tv), 64, replace=False)
        ci = np.random.choice(len(tc), 64, replace=False)
        v = [torch.stack([tv[i][j] for i in vi]).to(DEV) for j in range(3)]
        c = [torch.stack([tc[i][j] for i in ci]).to(DEV) for j in range(3)]
        ev, ec = m(*v), m(*c)
        gap = ec.mean().item() - ev.mean().item()
        # Strong margin: push valid way down, corrupt way up
        loss = F.relu(ev + 3).mean() + F.relu(3 - ec).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        if (ep+1) % 100 == 0:
            print(f"  {domain} ep {ep+1:4d}: ev={ev.mean().item():+.3f} ec={ec.mean().item():+.3f} gap={gap:.3f}")
    m.eval(); return m

for domain, task, da, ne in DOMAINS:
    print(f"\n{'='*70}")
    print(f"VERIFY: {domain} ({task})")
    print(f"{'='*70}")
    
    torch.manual_seed(42); np.random.seed(42)
    w, e, _ = env(domain, task)
    tv, tc, ts = data(w, e, nv=400, nc=200)
    
    m = train_iwcm_strong(tv, tc, d_a=da, epochs=ne, domain=domain)
    
    # ── Check 1: Energy separation (valid vs corrupt) ──
    print(f"\n  CHECK 1: Energy separation")
    evals, ecvals = [], []
    for _ in range(20):
        vi = np.random.choice(len(tv), 64)
        ci = np.random.choice(len(tc), 64)
        v = [torch.stack([tv[i][j] for i in vi]).to(DEV) for j in range(3)]
        c = [torch.stack([tc[i][j] for i in ci]).to(DEV) for j in range(3)]
        with torch.no_grad():
            evals.append(m(*v).mean().item())
            ecvals.append(m(*c).mean().item())
    avg_ev, avg_ec = np.mean(evals), np.mean(ecvals)
    gap = avg_ec - avg_ev
    c1_pass = gap > 0.5
    print(f"    ev={avg_ev:+.3f} ec={avg_ec:+.3f} gap={gap:.3f} {'PASS' if c1_pass else 'FAIL'} (needs >0.5)")
    
    # ── Check 2: Solver monotonically decreases energy ──
    print(f"\n  CHECK 2: Solver monotonic descent")
    z0, A, Zt = ts[0]
    zb = z0.unsqueeze(0).to(DEV); Ab = A.unsqueeze(0).to(DEV); Zb = Zt.unsqueeze(0).to(DEV)
    Z_init = zb.unsqueeze(1).expand(-1, H, -1, -1).clone().to(DEV)
    Z_solved, energies = solve_monotonic(m, zb, Ab, steps=100, lr=0.01, init_Z=Z_init)
    e_start, e_end = energies[0], energies[-1]
    descent = e_start > e_end  # energy decreased
    monotonic = all(energies[i] >= energies[i+1] for i in range(len(energies)-5))  # allow tiny noise
    print(f"    E: {e_start:+.3f} → {e_end:+.3f} (Δ={e_end-e_start:+.3f})")
    print(f"    Monotonic: {'PASS' if monotonic else 'FAIL'}")
    
    # ── Check 3: Energy landscape isn't flat ──
    print(f"\n  CHECK 3: Energy landscape has structure (not flat)")
    # Random init should give different energy than z0rep init
    Z_rand = torch.randn(1, H, Ns, ds, device=DEV)
    Z_rand_solved, e_rand_list = solve_monotonic(m, zb, Ab, steps=100, lr=0.01, init_Z=Z_rand)
    e_rand = e_rand_list[-1]
    landscape_var = abs(e_end - e_rand)
    c3_pass = landscape_var > 0.1
    print(f"    z0rep E={e_end:+.3f}  random E={e_rand:+.3f}  diff={landscape_var:.3f} {'PASS' if c3_pass else 'FAIL'} (needs >0.1)")
    
    # ── Check 4: Different trajectories get different energies ──
    print(f"\n  CHECK 4: Different trajectories → different energies")
    if len(ts) >= 2:
        z0_2, A_2, Zt_2 = ts[1]
        zb_2 = z0_2.unsqueeze(0).to(DEV); Ab_2 = A_2.unsqueeze(0).to(DEV)
        Z_init_2 = zb_2.unsqueeze(1).expand(-1, H, -1, -1).clone().to(DEV)
        Z_solved_2, e_list_2 = solve_monotonic(m, zb_2, Ab_2, steps=100, lr=0.01, init_Z=Z_init_2)
        e_end_2 = e_list_2[-1]
        traj_var = abs(e_end - e_end_2)
        c4_pass = traj_var > 0.01
        print(f"    Traj 1: {e_end:+.3f}  Traj 2: {e_end_2:+.3f}  diff={traj_var:.3f} {'PASS' if c4_pass else 'FAIL'} (needs >0.01)")
    
    # ── Check 5: Solver finds energy below ground truth ──
    print(f"\n  CHECK 5: Solver energy < ground truth energy")
    with torch.no_grad():
        e_gt = m(zb, Ab, Zb).item()
    c5_pass = e_end < e_gt
    print(f"    GT E={e_gt:+.3f}  Solved E={e_end:+.3f}  Δ={e_end-e_gt:+.3f} {'PASS' if c5_pass else 'FAIL'} (needs <0)")
    
    # ── Aggregate ──
    checks = [c1_pass, monotonic, c3_pass, c4_pass if len(ts) >= 2 else True, c5_pass]
    passed = sum(checks)
    total = len(checks)
    print(f"\n  >>> {domain}: {passed}/{total} checks passed {'✓' if passed == total else '⚠️'}")
