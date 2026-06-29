#!/usr/bin/env python3
"""Profiler: find where solver time goes, then fix the bottleneck."""
import sys, torch, numpy as np, time
from pathlib import Path
BASE = str(Path(__file__).resolve().parent.parent.parent)
sys.path.insert(0, BASE)
torch.set_num_threads(1)

from scripts.experiments._common import *

DEV = torch.device('cuda')
dtype = torch.float32

w, e, da = env()
tv, tc, ts = data(w, e)
m = train_iwcm(tv, tc, d_a=da)
rl = Rollout(8, 19, da).to(DEV); train_rl(rl, tv, 100)

H, Ns, ds = 100, 8, 19

# Grab one test trajectory
z0, A, Zt = ts[0]
zb = z0.unsqueeze(0).to(DEV)
Ab = A.unsqueeze(0).to(DEV)
Zz_init = z0.unsqueeze(0).unsqueeze(1).expand(-1, H, -1, -1).clone().to(DEV)

# ── Profile each piece ────────────────────────────────────────────────────
torch.cuda.synchronize()
t0 = time.perf_counter()
for _ in range(100):
    _ = m(zb, Ab, Zz_init)
torch.cuda.synchronize()
t_fwd_100 = (time.perf_counter() - t0) / 100
print(f"Forward only (100x):    {t_fwd_100*1e6:.1f} μs each")

# One step with autograd
Z = Zz_init.clone().detach().requires_grad_(True)
vel = torch.zeros_like(Z)
torch.cuda.synchronize()
t0 = time.perf_counter()
for _ in range(50):
    loss = m(zb, Ab, Z).mean()
    g = torch.autograd.grad(loss, Z, create_graph=False)[0]
    vel = 0.9 * vel + g
    Z = Z.detach() - 0.01 * vel
    Z.requires_grad_(True); vel = vel.detach()
torch.cuda.synchronize()
t_step = (time.perf_counter() - t0) / 50
print(f"One solver step (fwd+bwd+update): {t_step*1e6:.1f} μs")

# Python loop overhead: empty loop of 100 iterations
t0 = time.perf_counter()
for _ in range(100000):
    pass
t_py_loop = (time.perf_counter() - t0) / 100000
print(f"Python loop overhead:   {t_py_loop*1e6:.3f} μs per iteration")
print(f"  100 iterations:       {t_py_loop*1e6*100:.1f} μs total")

# ── Try: torch.func.grad (functional gradient, no graph building) ─────────
from torch.func import grad as func_grad

def energy_loss(Z):
    return m(zb, Ab, Z).mean()

# Compile the gradient function
grad_fn = func_grad(energy_loss)

# Warmup
Zg = Zz_init.clone().detach().requires_grad_(True)
g = grad_fn(Zg)
torch.cuda.synchronize()

# Timed
Zg = Zz_init.clone().detach().requires_grad_(True)
torch.cuda.synchronize()
t0 = time.perf_counter()
for _ in range(50):
    g = grad_fn(Zg)
    Zg = Zg.detach() - 0.01 * (0.9 * torch.zeros_like(Zg) + g)
    Zg.requires_grad_(True)
torch.cuda.synchronize()
t_func = (time.perf_counter() - t0) / 50
print(f"\ntorch.func.grad step:   {t_func*1e6:.1f} μs (vs {t_step*1e6:.1f} μs autograd)")

# ── Try: compiled gradient function ───────────────────────────────────────
@torch.compile(fullgraph=True)
def compiled_grad(Z):
    return torch.autograd.grad(m(zb, Ab, Z).mean(), Z, create_graph=False)[0]

Zg = Zz_init.clone().detach().requires_grad_(True)
g = compiled_grad(Zg)
torch.cuda.synchronize()

Zg = Zz_init.clone().detach().requires_grad_(True)
torch.cuda.synchronize()
t0 = time.perf_counter()
for _ in range(50):
    g = compiled_grad(Zg)
    Zg = Zg.detach() - 0.01 * (0.9 * torch.zeros_like(Zg) + g)
    Zg.requires_grad_(True)
torch.cuda.synchronize()
t_comp = (time.perf_counter() - t0) / 50
print(f"Compiled grad step:     {t_comp*1e6:.1f} μs")

# ── Try: vjp (vector-Jacobian product) ────────────────────────────────────
from torch.func import vjp

def energy_val(Z):
    return m(zb, Ab, Z).mean()

Zg = Zz_init.clone().detach().requires_grad_(True)
val, vjp_fn = vjp(energy_val, Zg)
torch.cuda.synchronize()
t0 = time.perf_counter()
for _ in range(50):
    val, vjp_fn = vjp(energy_val, Zg)
    g, = vjp_fn(torch.ones_like(val))
    Zg = Zg.detach() - 0.01 * (0.9 * torch.zeros_like(Zg) + g)
    Zg.requires_grad_(True)
torch.cuda.synchronize()
t_vjp = (time.perf_counter() - t0) / 50
print(f"torch.func.vjp step:    {t_vjp*1e6:.1f} μs")

# ── Try: forward-mode AD with jvp ─────────────────────────────────────────
from torch.func import jvp

Zg = Zz_init.clone().detach().requires_grad_(True)
torch.cuda.synchronize()
t0 = time.perf_counter()
for _ in range(50):
    # Use forward-mode: cheap if output < input dim (scalar loss, large Z)
    val, jvp_fn = jvp(lambda x: m(zb, Ab, x).mean(), (Zg,), (torch.ones_like(Zg),))
    # Actually jvp gives directional derivative, what we want is vjp (reverse mode)
    pass
torch.cuda.synchronize()
t_jvp = (time.perf_counter() - t0) / 50
print(f"jvp setup:              {t_jvp*1e6:.1f} μs")

# ── Try: manual step (no autograd) using precomputed gradient formula ─────
# For the energy function: E = sum over slots of lambda * agg(head(pool(shared(Z))))
# The gradient can be approximated by finite differences for speed
# But this is a last resort.

print(f"\n{'='*60}")
print("Bottleneck analysis:")
print(f"  Forward:  {t_fwd_100*1e6:.1f} μs")
print(f"  Step:     {t_step*1e6:.1f} μs")
print(f"  Pure bwd: {(t_step - t_fwd_100)*1e6:.1f} μs (bwd + update)")
print(f"  100 steps: {t_step*1e6*100:.1f} μs = {t_step*100*1e3:.1f} ms")

# ── Check: enable TF32 (faster matmul, acceptable for solver) ─────────────
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

Zg = Zz_init.clone().detach().requires_grad_(True)
vel = torch.zeros_like(Zg)
torch.cuda.synchronize()
t0 = time.perf_counter()
for _ in range(50):
    loss = m(zb, Ab, Zg).mean()
    g = torch.autograd.grad(loss, Zg, create_graph=False)[0]
    vel = 0.9 * vel + g
    Zg = Zg.detach() - 0.01 * vel
    Zg.requires_grad_(True); vel = vel.detach()
torch.cuda.synchronize()
t_tf32 = (time.perf_counter() - t0) / 50
print(f"\nTF32 enabled step:      {t_tf32*1e6:.1f} μs (vs {t_step*1e6:.1f} μs no TF32)")

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

# ── Try: reduce precision for gradient computation only ───────────────────
# Keep Z in fp32 but use fp16 for energy function forward
# This may or may not work depending on numerical sensitivity

print(f"\n{'─'*60}")
print("Quick wins summary:")
fwd_us = t_fwd_100 * 1e6
step_us = t_step * 1e6
print(f"  Forward: {fwd_us:.1f}μs ({fwd_us/step_us*100:.0f}% of step)")
print(f"  Backward + update: {step_us - fwd_us:.1f}μs ({(step_us-fwd_us)/step_us*100:.0f}% of step)")
print(f"  Python overhead (100 iters): {t_py_loop*1e6*100:.1f}μs ({(t_py_loop*100)/t_step*100:.0f}% of step)")
print(f"  TF32 speedup: {t_step/t_tf32:.2f}×")
