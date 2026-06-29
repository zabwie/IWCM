#!/usr/bin/env python3
"""DINO ViT-S/14 → pixel slots → IWCM flat-energy-across-H test."""

import torch, torch.nn as nn, torch.nn.functional as F
import torchvision.models as models, numpy as np, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.env.dm_control_wrapper import DMControlWrapper
from src.iwcm.fused_energy import FusedIWCMEnergy
from sklearn.metrics import roc_auc_score

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
S, D, LR, EPOCHS = 2, 64, 1e-3, 300
H_TRAIN = 25
H_TEST = [10, 25, 50, 100]
N_TRAIN, N_TEST = 200, 50
SOLVER_STEPS, SOLVER_LR = 100, 0.05

# ─── Models ───────────────────────────────────────────────────────────────────
print("Loading DINO ViT-S/14...", end=" ", flush=True)
dino = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14').to(DEVICE)
dino.eval().requires_grad_(False)
print(f"{sum(p.numel() for p in dino.parameters())/1e6:.1f}M params")

# Projector: patch-token mean pool → 384-dim → 2×64 slots
proj = nn.Linear(384, S * D).to(DEVICE)  # 49K params
energy = FusedIWCMEnergy(D, 1, S).to(DEVICE)
params = list(proj.parameters()) + list(energy.parameters())
opt = torch.optim.Adam(params, lr=LR)

def frames_to_slots(frames):
    """(B, H, 64, 64, 3) uint8 → (B, H, S, D) via DINO. Batched to avoid OOM."""
    B, H_ = frames.shape[:2]
    x = frames.reshape(B * H_, 64, 64, 3).permute(0, 3, 1, 2).float() / 255.0
    x = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
    s_list = []
    bs = 16  # frames per batch to avoid OOM
    for i in range(0, len(x), bs):
        batch = x[i:i+bs].to(DEVICE)
        with torch.no_grad():
            tokens = dino.forward_features(batch)['x_norm_patchtokens']
            f = tokens.mean(dim=1)  # (bs, 384)
            s_list.append(proj(f).cpu())
    s = torch.cat(s_list, dim=0)
    return s.reshape(B, H_, S, D).to(DEVICE)

# ─── Data ──────────────────────────────────────────────────────────────────────
def gen_traj(valid, rng, horizon=H_TRAIN):
    wrapper = DMControlWrapper('cartpole', 'swingup')
    env = wrapper._env
    env.reset()
    z0_frame = env.physics.render(camera_id=0, height=64, width=64)
    frames, acts = [], []
    corrupt_step = None if valid else rng.randint(horizon // 4, 3 * horizon // 4)
    for t in range(horizon):
        a = wrapper.sample_action(); acts.append(a)
        ts = env.step(a)
        if t == corrupt_step:
            ct = rng.choice(['teleport', 'freeze', 'reverse'])
            if ct == 'teleport':
                env.physics.data.qpos += rng.randn(*env.physics.data.qpos.shape) * 0.2
            elif ct == 'freeze':
                env.physics.data.qvel[:] = 0
            elif ct == 'reverse':
                env.physics.data.qvel *= -1.5
        frames.append(env.physics.render(camera_id=0, height=64, width=64))
        if ts.last() and t < horizon - 1:
            return None
    return z0_frame, np.stack(frames), np.stack(acts)

def collect(n, valid, rng, horizon=H_TRAIN):
    data = []
    for _ in range(n * 10):
        r = gen_traj(valid, rng, horizon)
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
    return z0.to(DEVICE), A.to(DEVICE), Z.to(DEVICE)

# ─── Train ─────────────────────────────────────────────────────────────────────
print(f"DINO IWCM — D={D}, H={H_TRAIN}")
t0 = time.time()
train_v = collect(N_TRAIN, True, np.random.RandomState(42))
train_c = collect(N_TRAIN, False, np.random.RandomState(43))
vz0, vA, vZ = encode(train_v)
cz0, cA, cZ = encode(train_c)
test_v = collect(N_TEST, True, np.random.RandomState(44))
test_c = collect(N_TEST, False, np.random.RandomState(45))
tz0, tA, tZ = encode(test_v + test_c)
t_labels = torch.cat([torch.zeros(N_TEST), torch.ones(N_TEST)]).numpy()
print(f"  Data: {time.time()-t0:.1f}s. Train: {len(train_v)}v+{len(train_c)}c")

for ep in range(EPOCHS + 1):
    vi = np.random.choice(len(train_v), 32, replace=False)
    ci = np.random.choice(len(train_c), 32, replace=False)
    opt.zero_grad()
    ev = energy(vz0[vi], vA[vi], vZ[vi])
    ec = energy(cz0[ci], cA[ci], cZ[ci])
    loss = F.relu(ev + 1.0).mean() + F.relu(1.0 - ec).mean() + 1e-3 * (ev.pow(2).mean() + ec.pow(2).mean())
    loss.backward()
    nn.utils.clip_grad_norm_(params, 1.0)
    opt.step()
    if ep % 100 == 0:
        print(f"  ep {ep:4d}: ev={ev.mean():+.3f} ec={ec.mean():+.3f} loss={loss.item():.4f}")

energy.eval(); proj.eval()
with torch.no_grad():
    e_test = energy(tz0, tA, tZ).cpu().numpy()
auroc = roc_auc_score(t_labels, e_test)
e_v, e_c = e_test[t_labels == 0], e_test[t_labels == 1]
print(f"\n  AUROC: {auroc:.3f}  (E_valid={e_v.mean():+.3f}  E_corrupt={e_c.mean():+.3f})")

# ─── Flat-energy test ──────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"Flat-energy-across-H (DINO slots, z₀-replication solver)")
print(f"{'='*60}")
print(f"{'H':>8}  {'E_before':>10}  {'E_after':>10}  {'ΔE':>10}  {'ground_truth':>14}")
print(f"{'─'*8}  {'─'*10}  {'─'*10}  {'─'*10}  {'─'*14}")

wrapper = DMControlWrapper('cartpole', 'swingup')
env = wrapper._env
rng = np.random.RandomState(99)
max_H = max(H_TEST)

traj = None
for _ in range(100):
    r = gen_traj(True, rng, horizon=max_H)
    if r is not None: traj = r; break

assert traj is not None

z0f = torch.from_numpy(traj[0][None, None].copy()).float()
zf = torch.from_numpy(traj[1][None].copy()).float()
A_full = torch.from_numpy(traj[2].copy()).float()

with torch.no_grad():
    z0 = frames_to_slots(z0f)[:, 0]
    Z_full = frames_to_slots(zf)

results = []
for H_target in H_TEST:
    Z_t = Z_full[:, :H_target].clone()
    A_t = A_full[:H_target].unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        Z_init = z0.unsqueeze(1).expand(-1, H_target, -1, -1).clone()
        e_before = energy(z0, A_t, Z_init).item()

    Z_sol = Z_init.clone().detach().requires_grad_(True)
    solver_opt = torch.optim.SGD([Z_sol], lr=SOLVER_LR, momentum=0.9)

    for _ in range(SOLVER_STEPS):
        solver_opt.zero_grad()
        e = energy(z0, A_t, Z_sol)
        e.backward()
        solver_opt.step()

    with torch.no_grad():
        e_after = energy(z0, A_t, Z_sol).item()
        e_ground_truth = energy(z0, A_t, Z_t.to(DEVICE)).item()

    results.append((H_target, e_before, e_after, e_ground_truth))
    print(f"  {H_target:>4}H  {e_before:>+10.3f}  {e_after:>+10.3f}  {e_after-e_before:>+10.3f}  {e_ground_truth:>+14.3f}")

std_flat = np.std([r[2] for r in results])
print(f"\n  Energy std across H: {std_flat:.3f}")
print(f"  {'✓ FLAT — drift eliminated' if std_flat < 1.0 else '⚠ NOT FLAT'}")

torch.save({'energy': energy.state_dict(), 'proj': proj.state_dict(),
            'auroc': auroc, 'results': results, 'backbone': 'dinov2_vits14'},
           Path(__file__).parent / 'flat_energy_dino_results.pt')
print(f"\n  Results saved to scripts/flat_energy_dino_results.pt")
