#!/usr/bin/env python3
"""Frozen ResNet → pixel slots → IWCM on cartpole. (spatial v2 — 4x4 features)

Backbone: ResNet-18 up to layer3 (frozen) → (256, 4, 4) features per frame.
          Higher spatial resolution than layer4's 2x2 — corruption detection needs it.
Usage:
    python scripts/pixel_train.py
    python scripts/pixel_train.py --multi-seed
"""

import torch, torch.nn as nn, torch.nn.functional as F
import torchvision.models as models, numpy as np, sys, time, argparse
from pathlib import Path
from sklearn.metrics import roc_auc_score
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.env.dm_control_wrapper import DMControlWrapper
from src.iwcm.fused_energy import FusedIWCMEnergy

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
S, D = 2, 64
H_TRAIN = 25
H_TEST = [10, 25, 50, 100]
SOLVER_STEPS, SOLVER_LR = 100, 0.05

rn18 = models.resnet18(weights='DEFAULT')
bb = nn.Sequential(*list(rn18.children())[:7])
bb.eval().requires_grad_(False)
pool = nn.AdaptiveAvgPool2d(1)
proj = nn.Linear(256, S * D)
energy = FusedIWCMEnergy(D, 1, S)

def frames_to_slots(frames):
    B, H_ = frames.shape[:2]
    x = frames.reshape(B * H_, 64, 64, 3).permute(0, 3, 1, 2).float() / 255.0
    s_list = []
    for i in range(0, len(x), 256):
        f = bb(x[i:i+256].to(DEVICE))
        s_list.append(proj(pool(f).reshape(f.size(0), -1)).cpu())
    return torch.cat(s_list, dim=0).reshape(B, H_, S, D).to(DEVICE)

def gen_traj(wrapper, env, valid, rng, horizon):
    env.reset()
    z0f = env.physics.render(camera_id=0, height=64, width=64)
    frames, acts = [], []
    cs = None if valid else rng.randint(horizon // 4, 3 * horizon // 4)
    for t in range(horizon):
        a = wrapper.sample_action(); acts.append(a)
        ts = env.step(a)
        if t == cs:
            ct = rng.choice(['teleport', 'freeze', 'reverse'])
            if ct == 'teleport':
                env.physics.data.qpos += rng.randn(*env.physics.data.qpos.shape) * 0.2
            elif ct == 'freeze':
                env.physics.data.qvel[:] = 0
            elif ct == 'reverse':
                env.physics.data.qvel *= -1.5
        frames.append(env.physics.render(camera_id=0, height=64, width=64))
        if ts.last() and t < horizon - 1: return None
    return z0f, np.stack(frames), np.stack(acts)

def collect(n, valid, rng, horizon=H_TRAIN):
    wrapper = DMControlWrapper('cartpole', 'swingup'); env = wrapper._env
    data = []
    for _ in range(n * 10):
        r = gen_traj(wrapper, env, valid, rng, horizon)
        if r is None: continue
        data.append(r)
        if len(data) >= n: break
    return data

def encode(data):
    z0f = np.stack([d[0] for d in data])
    zf = np.stack([d[1] for d in data])
    A = torch.stack([torch.from_numpy(d[2]) for d in data]).float()
    with torch.no_grad():
        z0 = frames_to_slots(torch.from_numpy(z0f[:, None]).float())[:, 0]
        Z = frames_to_slots(torch.from_numpy(zf).float())
    return z0, A.to(DEVICE), Z

def run_seed(seed, n_train, epochs, quick=False):
    print(f"\n--- Seed {seed} ---")
    np.random.seed(seed); torch.manual_seed(seed)
    bb.to(DEVICE); pool.to(DEVICE); proj.to(DEVICE); energy.to(DEVICE)
    opt = torch.optim.Adam(list(proj.parameters()) + list(energy.parameters()), lr=1e-3)

    t0 = time.time()
    train_v = collect(n_train, True, np.random.RandomState(seed))
    train_c = collect(n_train, False, np.random.RandomState(seed + 1))
    vz0, vA, vZ = encode(train_v); cz0, cA, cZ = encode(train_c)
    n_test = min(50, n_train // 4)
    test_v = collect(n_test, True, np.random.RandomState(seed + 2))
    test_c = collect(n_test, False, np.random.RandomState(seed + 3))
    tz0, tA, tZ = encode(test_v + test_c)
    t_labels = torch.cat([torch.zeros(n_test), torch.ones(n_test)])
    print(f"  Data: {time.time()-t0:.1f}s ({len(train_v)}v+{len(train_c)}c)")

    for ep in range(1, epochs + 1):
        vi = np.random.choice(len(train_v), 32, replace=False)
        ci = np.random.choice(len(train_c), 32, replace=False)
        opt.zero_grad()
        ev = energy(vz0[vi], vA[vi], vZ[vi])
        ec = energy(cz0[ci], cA[ci], cZ[ci])
        loss = F.relu(ev+1).mean() + F.relu(1-ec).mean() + 1e-3*(ev.pow(2).mean()+ec.pow(2).mean())
        loss.backward()
        nn.utils.clip_grad_norm_(opt.param_groups[0]['params'], 1.0)
        opt.step()
        if ep % 100 == 0 or quick:
            with torch.no_grad():
                auroc = roc_auc_score(t_labels.numpy(), energy(tz0, tA, tZ).cpu().numpy())
            print(f"  ep {ep:4d}: ev={ev.mean():+.3f} ec={ec.mean():+.3f} loss={loss:.4f} AUROC={auroc:.3f}")

    with torch.no_grad():
        auroc = roc_auc_score(t_labels.numpy(), energy(tz0, tA, tZ).cpu().numpy())
    print(f"  Final AUROC: {auroc:.3f}")

    if not quick:
        wrapper = DMControlWrapper('cartpole', 'swingup'); env = wrapper._env
        rng = np.random.RandomState(99)
        traj = None
        for _ in range(100):
            r = gen_traj(wrapper, env, True, rng, horizon=max(H_TEST))
            if r is not None: traj = r; break
        if traj:
            z0f = torch.from_numpy(traj[0][None, None].copy()).float()
            zf = torch.from_numpy(traj[1][None].copy()).float()
            A_full = torch.from_numpy(traj[2].copy()).float()
            with torch.no_grad():
                z0 = frames_to_slots(z0f)[:, 0]; Z_full = frames_to_slots(zf)
            es = []
            for Ht in H_TEST:
                Z_t = Z_full[:, :Ht].clone(); A_t = A_full[:Ht].unsqueeze(0).to(DEVICE)
                Z_sol = z0.unsqueeze(1).expand(-1, Ht, -1, -1).clone().detach().requires_grad_(True)
                opt_s = torch.optim.SGD([Z_sol], lr=SOLVER_LR, momentum=0.9)
                for _ in range(SOLVER_STEPS):
                    opt_s.zero_grad(); e = energy(z0, A_t, Z_sol); e.backward(); opt_s.step()
                with torch.no_grad():
                    es.append(energy(z0, A_t, Z_sol).item())
            std = np.std(es)
            print(f"  Flat-energy: std={std:.3f} {'FLAT' if std < 1.0 else 'NOT FLAT'}")
            for i, Ht in enumerate(H_TEST):
                print(f"    {Ht:4d}H: E={es[i]:+.3f}")
    return auroc

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--quick', action='store_true')
    parser.add_argument('--multi-seed', action='store_true')
    args = parser.parse_args()
    epochs = 50 if args.quick else 300
    n_train = 50 if args.quick else 500

    if args.multi_seed:
        aurocs = [run_seed(s, n_train, epochs, args.quick) for s in [42, 43, 44]]
        print(f"\nAUROC: {np.mean(aurocs):.3f} +- {np.std(aurocs):.3f}")
    else:
        run_seed(42, n_train, epochs, args.quick)
