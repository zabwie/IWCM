#!/usr/bin/env python3
"""Investigate the amortized solver ΔE gap across domains.

Cartpole: ΔE = -0.83 (fine)
Cheetah:  ΔE = -4.51 (huge negative gap — amortized >> exact K20?)
Walker:   ΔE = -0.03 (perfect match)

Hypotheses:
  1. K20 exact solver doesn't converge well on cheetah → amortized generalizes better
  2. Training data quantity/quality differs per domain
  3. Energy landscape is differently shaped
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

def solve_exact(m, z0, A, steps, lr, init_Z):
    Z = init_Z.clone().detach().requires_grad_(True); vel = torch.zeros_like(Z)
    for _ in range(steps):
        loss = m(z0, A, Z).mean()
        g = torch.autograd.grad(loss, Z, create_graph=False)[0]
        vel = 0.9 * vel + g
        Z = Z.detach() - lr * vel
        Z.requires_grad_(True); vel = vel.detach()
    return Z

DOMAINS = [('cartpole', 'swingup'), ('cheetah', 'run'), ('walker', 'walk')]

for domain, task in DOMAINS:
    print(f"\n{'='*70}")
    print(f"DOMAIN: {domain}/{task}")
    print(f"{'='*70}")

    torch.manual_seed(42)
    np.random.seed(42)

    w, e, da = env(name=domain, task=task)
    tv, tc, ts = data(w, e, nv=400, nc=200)
    print(f"  Data: {len(tv)} train valid, {len(tc)} train corrupt, {len(ts)} test")

    m = train_iwcm(tv, tc, d_a=da)
    print(f"  IWCM energy trained: {sum(p.numel() for p in m.parameters()):,} params")

    # ── Check exact K20 solver convergence ──
    print(f"\n  ── K20 solver diagnostics ──")
    k20_de_vs_true = []
    k20_mse_vs_true = []
    k20_energy_true = []
    k20_energy_solved = []

    for z0, A, Zt in ts[:20]:
        zb = z0.unsqueeze(0).to(DEV); Ab = A.unsqueeze(0).to(DEV); Zb = Zt.unsqueeze(0).to(DEV)
        Z_init = z0.unsqueeze(0).unsqueeze(1).expand(-1, H, -1, -1).clone().to(DEV)

        with torch.no_grad():
            e_true = m(zb, Ab, Zb).item()

        # K20 solver
        Z_k20 = solve_exact(m, zb, Ab, 20, 0.08, Z_init)
        with torch.no_grad():
            e_k20 = m(zb, Ab, Z_k20).item()
            mse_k20 = F.mse_loss(Z_k20[0,:,:,PHYS], Zb[0,:,:,PHYS]).item()

        k20_de_vs_true.append(e_k20 - e_true)
        k20_mse_vs_true.append(mse_k20)
        k20_energy_true.append(e_true)
        k20_energy_solved.append(e_k20)

    print(f"  Ground truth energy: μ={np.mean(k20_energy_true):+.3f} σ={np.std(k20_energy_true):.3f}")
    print(f"  K20 solved energy:   μ={np.mean(k20_energy_solved):+.3f} σ={np.std(k20_energy_solved):.3f}")
    print(f"  K20 ΔE vs true:      μ={np.mean(k20_de_vs_true):+.3f} σ={np.std(k20_de_vs_true):.3f}")
    print(f"  K20 MSE vs true:     μ={np.mean(k20_mse_vs_true):.6f}")

    # ── K100 reference (gold standard) ──
    print(f"\n  ── K100 reference ──")
    k100_de_vs_true = []
    k100_de_vs_k20 = []

    for z0, A, Zt in ts[:20]:
        zb = z0.unsqueeze(0).to(DEV); Ab = A.unsqueeze(0).to(DEV); Zb = Zt.unsqueeze(0).to(DEV)
        Z_init = z0.unsqueeze(0).unsqueeze(1).expand(-1, H, -1, -1).clone().to(DEV)
        Z_k100 = solve_exact(m, zb, Ab, 100, 0.01, Z_init)
        Z_k20 = solve_exact(m, zb, Ab, 20, 0.08, Z_init)
        with torch.no_grad():
            e100 = m(zb, Ab, Z_k100).item()
            e20 = m(zb, Ab, Z_k20).item()
            etrue = m(zb, Ab, Zb).item()
        k100_de_vs_true.append(e100 - etrue)
        k100_de_vs_k20.append(e100 - e20)

    print(f"  K100 ΔE vs true: μ={np.mean(k100_de_vs_true):+.3f}")
    print(f"  K100 ΔE vs K20:  μ={np.mean(k100_de_vs_k20):+.3f}")
    print(f"  K20 {'CONVERGED' if abs(np.mean(k100_de_vs_k20)) < 0.5 else 'NOT CONVERGED'} (threshold |ΔE|<0.5 vs K100)")

    # ── Build K20 training data for all trajectories ──
    print(f"\n  ── Building K20 training data ──")
    X_z0, X_A, Y = [], [], []
    for z0, A, Zt in ts:
        zb = z0.unsqueeze(0).to(DEV); Ab = A.unsqueeze(0).to(DEV)
        Z_init = z0.unsqueeze(0).unsqueeze(1).expand(-1, H, -1, -1).clone().to(DEV)
        with torch.enable_grad():
            Z_k20 = solve_exact(m, zb, Ab, 20, 0.08, Z_init)
        X_z0.append(zb); X_A.append(Ab); Y.append(Z_k20)
    n_train = min(60, len(ts))

    # ── Train amortized solver ──
    print(f"\n  ── Training amortized solver ({n_train} trajectories) ──")
    net = torch.nn.Sequential(
        torch.nn.Linear(ds + da + 1, 256), torch.nn.ReLU(),
        torch.nn.Linear(256, 256), torch.nn.ReLU(),
        torch.nn.Linear(256, ds),
    ).to(DEV)

    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    for ep in range(200):
        losses = []
        for i in range(n_train):
            zb, Ab, Yt = X_z0[i], X_A[i], Y[i]
            z0_rep = zb.unsqueeze(1).expand(-1, H, -1, -1)
            time_enc = torch.linspace(0, 1, H, device=DEV).reshape(1, H, 1, 1).expand(-1, -1, Ns, -1)
            inp = torch.cat([z0_rep, Ab.unsqueeze(2).expand(-1, -1, Ns, -1), time_enc], dim=-1)
            Z_pred = z0_rep + net(inp)
            loss = F.mse_loss(Z_pred, Yt)
            with torch.no_grad():
                e_pred = m(zb, Ab, Z_pred).mean()
                e_tgt = m(zb, Ab, Yt).mean()
            (loss + 0.1 * torch.relu(e_pred - e_tgt + 0.01)).backward()
            opt.step(); opt.zero_grad()
            losses.append(loss.item())

    # ── Detailed evaluation ──
    print(f"\n  ── Amortized solver evaluation (test set) ──")
    eval_times, e_diffs, mses = [], [], []
    e_preds, e_tgts = [], []

    eval_indices = list(range(n_train, min(n_train + 10, len(ts))))
    for idx in eval_indices:
        zb, Ab, Yt = X_z0[idx], X_A[idx], Y[idx]
        Zb = ts[idx][2].unsqueeze(0).to(DEV)
        z0_rep = zb.unsqueeze(1).expand(-1, H, -1, -1)
        time_enc = torch.linspace(0, 1, H, device=DEV).reshape(1, H, 1, 1).expand(-1, -1, Ns, -1)
        with torch.no_grad():
            inp = torch.cat([z0_rep, Ab.unsqueeze(2).expand(-1, -1, Ns, -1), time_enc], dim=-1)
            Z_pred = z0_rep + net(inp)
            e_pred = m(zb, Ab, Z_pred).item()
            e_tgt = m(zb, Ab, Yt).item()
            mse_phys = F.mse_loss(Z_pred[0,:,:,PHYS], Zb[0,:,:,PHYS]).item()

        e_diffs.append(e_pred - e_tgt)
        e_preds.append(e_pred)
        e_tgts.append(e_tgt)
        mses.append(mse_phys)

    print(f"  Amortized ΔE vs K20:  μ={np.mean(e_diffs):+.3f} σ={np.std(e_diffs):.3f}")
    print(f"  Amortized E:          μ={np.mean(e_preds):+.3f}")
    print(f"  K20 target E:         μ={np.mean(e_tgts):+.3f}")
    print(f"  Amortized MSE vs GT:  μ={np.mean(mses):.6f}")

    # ── Sanity check: K20 vs K100 vs amortized head-to-head ──
    print(f"\n  ── Head-to-head on first test trajectory ──")
    first_eval = eval_indices[0]
    zb = X_z0[first_eval].to(DEV); Ab = X_A[first_eval].to(DEV); Zb = ts[first_eval][2].unsqueeze(0).to(DEV)
    Z_init = zb.unsqueeze(1).expand(-1, H, -1, -1).clone().to(DEV)

    Z_k100 = solve_exact(m, zb, Ab, 100, 0.01, Z_init)
    Z_k20 = solve_exact(m, zb, Ab, 20, 0.08, Z_init)
    z0_rep = zb.unsqueeze(1).expand(-1, H, -1, -1)
    time_enc = torch.linspace(0, 1, H, device=DEV).reshape(1, H, 1, 1).expand(-1, -1, Ns, -1)
    with torch.no_grad():
        Z_amort = z0_rep + net(torch.cat([z0_rep, Ab.unsqueeze(2).expand(-1, -1, Ns, -1), time_enc], dim=-1))
        e100 = m(zb, Ab, Z_k100).item()
        e20 = m(zb, Ab, Z_k20).item()
        e_am = m(zb, Ab, Z_amort).item()
        e_gt = m(zb, Ab, Zb).item()

        mse100 = F.mse_loss(Z_k100[0,:,:,PHYS], Zb[0,:,:,PHYS]).item()
        mse20 = F.mse_loss(Z_k20[0,:,:,PHYS], Zb[0,:,:,PHYS]).item()
        mse_am = F.mse_loss(Z_amort[0,:,:,PHYS], Zb[0,:,:,PHYS]).item()

    print(f"  Ground truth E={e_gt:+.3f}")
    print(f"  K100: E={e100:+.3f}  MSE={mse100:.6f}  ΔE_vs_gt={e100-e_gt:+.3f}")
    print(f"  K20:  E={e20:+.3f}  MSE={mse20:.6f}  ΔE_vs_gt={e20-e_gt:+.3f}  ΔE_vs_K100={e20-e100:+.3f}")
    print(f"  Amor: E={e_am:+.3f}  MSE={mse_am:.6f}  ΔE_vs_gt={e_am-e_gt:+.3f}  ΔE_vs_K20={e_am-e20:+.3f}")

print(f"\n{'='*70}")
print("DONE")
