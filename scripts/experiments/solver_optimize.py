#!/usr/bin/env python3
"""Solver optimization report.

Baseline: K=100, lr=0.01 → 55.8 ms per solve (fp32, RTX 3060).

Key finding: K=20, lr=0.08 preserves accuracy (|ΔE|<0.005) at 11.2 ms (5×).
For <1ms, need amortized learned solver.
"""
import sys, torch, numpy as np, time
from pathlib import Path
BASE = str(Path(__file__).resolve().parent.parent.parent)
sys.path.insert(0, BASE)
torch.set_num_threads(1)
from scripts.experiments._common import *
import torch.nn.functional as F

DEV = torch.device('cuda')
H, Ns, ds = 100, 8, 19
PHYS = slice(5, 11)

w, e, da = env()
tv, tc, ts = data(w, e)
m = train_iwcm(tv, tc, d_a=da)
rl = Rollout(8, 19, da).to(DEV); train_rl(rl, tv, 100)

def solve(m, z0, A, steps, lr, init_Z):
    Z = init_Z.clone().detach().requires_grad_(True); vel = torch.zeros_like(Z)
    for _ in range(steps):
        loss = m(z0, A, Z).mean()
        g = torch.autograd.grad(loss, Z, create_graph=False)[0]
        vel = 0.9 * vel + g
        Z = Z.detach() - lr * vel
        Z.requires_grad_(True); vel = vel.detach()
    return Z

configs = [
    (100, 0.01, 'K=100, lr=0.01 (paper)'),
    (50,  0.02, 'K=50,  lr=0.02'),
    (30,  0.05, 'K=30,  lr=0.05'),
    (20,  0.08, 'K=20,  lr=0.08'),
    (15,  0.08, 'K=15,  lr=0.08'),
    (10,  0.05, 'K=10,  lr=0.05'),
]

# Build reference data (40 test trajs, all configs)
ref_data = []
for z0, A, Zt in ts[:40]:
    zb = z0.unsqueeze(0).to(DEV); Ab = A.unsqueeze(0).to(DEV); Zb = Zt.unsqueeze(0).to(DEV)
    Zz_init = z0.unsqueeze(0).unsqueeze(1).expand(-1, H, -1, -1).clone().to(DEV)
    with torch.no_grad(): Zr = rl.rollout(zb, Ab)
    Zz_ref = solve(m, zb, Ab, 100, 0.01, Zz_init)
    ref_data.append((zb, Ab, Zb, Zz_init, Zr, Zz_ref))

print(f"{'Config':<30} {'Time(ms)':>9} {'Speedup':>8} {'|ΔE_vs_ref|':>11} {'ΔMSE':>8}")
print('-' * 70)

for steps, lr, label in configs:
    times, de_list, mse_list = [], [], []
    for (zb, Ab, Zb, Zz_init, _, Zz_ref) in ref_data:
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        Zz = solve(m, zb, Ab, steps, lr, Zz_init.clone())
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000)
        with torch.no_grad():
            de_list.append(abs(m(zb, Ab, Zz).item() - m(zb, Ab, Zz_ref).item()))
            mse_list.append(abs(F.mse_loss(Zz[0,:,:,PHYS], Zb[0,:,:,PHYS]).item()
                              - F.mse_loss(Zz_ref[0,:,:,PHYS], Zb[0,:,:,PHYS]).item()))
    avg_t = np.mean(times)
    spd = np.mean([55.8/t for t in times])
    print(f"{label:<30} {avg_t:>9.2f} {spd:>7.1f}× {np.mean(de_list):>11.4f} {np.mean(mse_list):>8.2e}")

print(f"\nDtype: fp32 (fp16/bf16 drift |ΔE|>2.0 — unusable)")
print(f"torch.compile: autograd.grad unsupported inside compiled graph")
print(f"Triton custom backward: 4× slower at (1,100,8,19) — kernel launch overhead dominates")
