#!/usr/bin/env python3
"""Retrain IWCM + amortized solver for all 3 domains with LayerNorm fix."""

import sys, torch, numpy as np, time
from pathlib import Path
BASE = str(Path(__file__).resolve().parent.parent.parent)
sys.path.insert(0, BASE)
torch.set_num_threads(1)
torch.manual_seed(42)
np.random.seed(42)

from scripts.experiments._common import *
import torch.nn.functional as F

DEV = torch.device('cuda')
H, Ns, ds = 100, 8, 19
PHYS = slice(5, 11)

DOMAINS = [('cartpole', 'swingup', 1), ('cheetah', 'run', 6), ('walker', 'walk', 6)]
N_EPOCHS = {'cartpole': 200, 'cheetah': 500, 'walker': 800}

def solve_exact(m, z0, A, steps, lr, init_Z):
    Z = init_Z.clone().detach().requires_grad_(True); vel = torch.zeros_like(Z)
    for _ in range(steps):
        loss = m(z0, A, Z).mean()
        g = torch.autograd.grad(loss, Z, create_graph=False)[0]
        vel = 0.9 * vel + g
        Z = Z.detach() - lr * vel
        Z.requires_grad_(True); vel = vel.detach()
    return Z

def train_iwcm_bce(tv, tc, d_a, epochs=300, domain=""):
    """BCE-based training — strong gradients everywhere."""
    m = FusedIWCMEnergy(ds, d_a, hidden=128, num_slots=Ns).to(DEV)
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    for ep in range(epochs):
        vi = np.random.choice(len(tv), 64, replace=False)
        ci = np.random.choice(len(tc), 64, replace=False)
        v = [torch.stack([tv[i][j] for i in vi]).to(DEV) for j in range(3)]
        c = [torch.stack([tc[i][j] for i in ci]).to(DEV) for j in range(3)]
        ev, ec = m(*v), m(*c)
        # BCE loss: p_valid = sigmoid(-ev), p_corrupt = sigmoid(-ec)
        pv = torch.sigmoid(-ev)
        pc = torch.sigmoid(-ec)
        loss = F.binary_cross_entropy(pv, torch.ones_like(pv)) + F.binary_cross_entropy(pc, torch.zeros_like(pc))
        opt.zero_grad(); loss.backward(); opt.step()
        if (ep+1) % 100 == 0:
            print(f"  {domain} ep {ep+1:4d}: ev={ev.mean().item():+.3f} ec={ec.mean().item():+.3f} loss={loss.item():.3f}")
    m.eval(); return m

for domain, task, da in DOMAINS:
    print(f"\n{'='*70}")
    print(f"{domain} ({task}, action_dim={da})")
    print(f"{'='*70}")

    w, e, _ = env(domain, task)
    tv, tc, ts = data(w, e, nv=400, nc=200)
    print(f"  Data: {len(tv)} valid, {len(tc)} corrupt, {len(ts)} test")

    ne = N_EPOCHS.get(domain, 300)
    m = train_iwcm_bce(tv, tc, d_a=da, epochs=ne, domain=domain)

    # ── Solver diagnostics ──
    print(f"\n  ── Solver diagnostics ──")
    gt_e, k20_e, k100_e = [], [], []
    k20_mse, k100_mse = [], []
    for z0, A, Zt in ts[:40]:
        zb = z0.unsqueeze(0).to(DEV); Ab = A.unsqueeze(0).to(DEV); Zb = Zt.unsqueeze(0).to(DEV)
        Z_init = zb.unsqueeze(1).expand(-1, H, -1, -1).clone().to(DEV)
        Zk20 = solve_exact(m, zb, Ab, 20, 0.08, Z_init)
        Zk100 = solve_exact(m, zb, Ab, 100, 0.01, Z_init)
        with torch.no_grad():
            gt_e.append(m(zb, Ab, Zb).item())
            k20_e.append(m(zb, Ab, Zk20).item())
            k100_e.append(m(zb, Ab, Zk100).item())
            k20_mse.append(F.mse_loss(Zk20[0,:,:,PHYS], Zb[0,:,:,PHYS]).item())
            k100_mse.append(F.mse_loss(Zk100[0,:,:,PHYS], Zb[0,:,:,PHYS]).item())

    print(f"  GT energy:        μ={np.mean(gt_e):+.3f} σ={np.std(gt_e):.3f}")
    print(f"  K20 energy:       μ={np.mean(k20_e):+.3f}")
    print(f"  K100 energy:      μ={np.mean(k100_e):+.3f}")
    print(f"  K20 ΔE vs GT:     μ={np.mean(np.array(k20_e)-np.array(gt_e)):+.3f}")
    print(f"  K100 ΔE vs GT:    μ={np.mean(np.array(k100_e)-np.array(gt_e)):+.3f}")
    print(f"  K20 ΔE vs K100:   μ={np.mean(np.array(k20_e)-np.array(k100_e)):+.3f}")
    print(f"  K20 MSE vs GT:    μ={np.mean(k20_mse):.6f}")
    print(f"  K100 MSE vs GT:   μ={np.mean(k100_mse):.6f}")
    k20_ok = abs(np.mean(np.array(k20_e)-np.array(k100_e))) < 0.5
    print(f"  K20 {'CONVERGED' if k20_ok else 'NOT CONVERGED'}")

    # ── Amortized solver ──
    print(f"\n  ── Amortized solver ──")
    X_z0, X_A, Y = [], [], []
    for z0, A, Zt in ts:
        zb = z0.unsqueeze(0).to(DEV); Ab = A.unsqueeze(0).to(DEV)
        Z_init = zb.unsqueeze(1).expand(-1, H, -1, -1).clone().to(DEV)
        with torch.enable_grad():
            Z_k20 = solve_exact(m, zb, Ab, 20, 0.08, Z_init)
        X_z0.append(zb); X_A.append(Ab); Y.append(Z_k20)

    n_train = min(60, len(ts))
    net = torch.nn.Sequential(
        torch.nn.Linear(ds + da + 1, 256), torch.nn.ReLU(),  # FIXED: ds+da+1 not ds+1+1
        torch.nn.Linear(256, 256), torch.nn.ReLU(),
        torch.nn.Linear(256, ds),
    ).to(DEV)

    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    for ep in range(200):
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

    # ── Evaluate ──
    eval_idx = list(range(n_train, min(n_train+10, len(ts))))
    e_diffs, mses, e_preds, e_tgts = [], [], [], []
    for idx in eval_idx:
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

    print(f"  Amort ΔE vs K20:  μ={np.mean(e_diffs):+.3f}")
    print(f"  Amort E:          μ={np.mean(e_preds):+.3f}")
    print(f"  K20 target E:     μ={np.mean(e_tgts):+.3f}")
    print(f"  Amort MSE vs GT:  μ={np.mean(mses):.6f}")
    print(f"  Amort ΔE vs GT:   μ={np.mean(np.array(e_preds)-np.array(gt_e[:len(e_preds)])):+.3f}")

print(f"\n{'='*70}")
print("DONE")
