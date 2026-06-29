"""Test: frame differencing as input to frozen ResNet + IWCM."""
import torch, torch.nn as nn, torch.nn.functional as F, numpy as np, sys, time
import torchvision.models as models
from sklearn.metrics import roc_auc_score
sys.path.insert(0, '.')
from src.env.dm_control_wrapper import DMControlWrapper
from src.iwcm.fused_energy import FusedIWCMEnergy

DEVICE = 'cuda'; H = 25; SEED = 42
torch.manual_seed(SEED); np.random.seed(SEED); rng = np.random.RandomState(SEED)

# Backbone: frozen ResNet layer3 (4x4 spatial)
bb = nn.Sequential(*list(models.resnet18(weights='DEFAULT').children())[:7])
bb.eval().requires_grad_(False).to(DEVICE)
pool = nn.AdaptiveAvgPool2d(1)
proj = nn.Linear(256, 128).to(DEVICE)  # 2 slots x 64 dim
energy = FusedIWCMEnergy(64, 1, 2).to(DEVICE)
opt = torch.optim.Adam(list(proj.parameters())+list(energy.parameters()), lr=1e-3)

wrapper = DMControlWrapper('cartpole', 'swingup'); env = wrapper._env

def gen(valid):
    env.reset()
    frames, acts = [], []
    cs = None if valid else rng.randint(H//4, 3*H//4)
    for t in range(H):
        a = wrapper.sample_action(); acts.append(a)
        ts = env.step(a)
        if t == cs:
            ct = rng.choice(['teleport','freeze','reverse'])
            if ct == 'teleport': env.physics.data.qpos += rng.randn(*env.physics.data.qpos.shape)*0.2
            elif ct == 'freeze': env.physics.data.qvel[:] = 0
            elif ct == 'reverse': env.physics.data.qvel *= -1.5
        frames.append(env.physics.render(camera_id=0, height=64, width=64))
        if ts.last() and t < H-1: return None
    return np.stack(frames), np.stack(acts)

def collect(n, valid):
    return [g for g in [gen(valid) for _ in range(n*10)] if g is not None][:n]

def compute_feats(frames):
    B, H_ = frames.shape[:2]
    x = torch.from_numpy(frames.reshape(-1, 64, 64, 3).copy()).permute(0, 3, 1, 2).float() / 255.0
    f = bb(x.to(DEVICE))
    return pool(f).reshape(B, H_, -1).cpu()

def frame_diff_feats(frames):
    B, H_ = frames.shape[:2]
    pairs = []
    for t in range(H_ - 1):
        diff = np.abs(frames[:, t+1].astype(np.float32) - frames[:, t].astype(np.float32))
        pairs.append(diff)
    diffs = np.stack(pairs, axis=1)
    x = torch.from_numpy(diffs.reshape(-1, 64, 64, 3).copy()).permute(0, 3, 1, 2).float() / 255.0
    f = bb(x.to(DEVICE))
    return pool(f).reshape(B, H_-1, -1).cpu()

print("Generating data...")
t0 = time.time()
tv = collect(500, True); tc = collect(500, False)
tev = collect(100, True); tec = collect(100, False)
print(f"  {time.time()-t0:.1f}s")

frames_v = np.stack([d[0] for d in tv])
acts_v = np.stack([d[1] for d in tv])
frames_c = np.stack([d[0] for d in tc])
acts_c = np.stack([d[1] for d in tc])
frames_te = np.stack([d[0] for d in tev+tec])
acts_te = np.stack([d[1] for d in tev+tec])
tl = torch.cat([torch.zeros(100), torch.ones(100)])

for mode_name, mode_fn, h_dim in [("Frame diff", frame_diff_feats, 24), ("Raw frames", compute_feats, 25)]:
    print(f"\n=== {mode_name} ===")
    proj = nn.Linear(256, 128).to(DEVICE)
    energy = FusedIWCMEnergy(64, 1, 2).to(DEVICE)
    opt = torch.optim.Adam(list(proj.parameters())+list(energy.parameters()), lr=1e-3)

    v_feats = mode_fn(frames_v)
    c_feats = mode_fn(frames_c)
    t_feats = mode_fn(frames_te)
    shift = 1 if mode_name == "Frame diff" else 0
    A_all = torch.from_numpy(acts_v[:, shift:]).float().to(DEVICE)
    cA_all = torch.from_numpy(acts_c[:, shift:]).float().to(DEVICE)
    tA = torch.from_numpy(acts_te[:, shift:]).float().to(DEVICE)

    for ep in range(301):
        vi=np.random.choice(500,64); ci=np.random.choice(500,64)
        zv = proj(v_feats[vi].to(DEVICE)).reshape(64, h_dim, 2, 64)
        zc = proj(c_feats[ci].to(DEVICE)).reshape(64, h_dim, 2, 64)
        Z = torch.cat([zv, zc]); A = torch.cat([A_all[vi], cA_all[ci]])
        opt.zero_grad()
        e = energy(torch.zeros(128,2,64,device=DEVICE), A, Z)
        ev, ec = e[:64].mean(), e[64:].mean()
        loss = F.relu(ev+1) + F.relu(1-ec) + 1e-3*(ev**2+ec**2)
        loss.backward(); nn.utils.clip_grad_norm_(opt.param_groups[0]['params'],1.0); opt.step()
        if ep%100==0:
            with torch.no_grad():
                tZ = proj(t_feats.to(DEVICE)).reshape(200, h_dim, 2, 64)
                au = roc_auc_score(tl.numpy(), energy(torch.zeros(200,2,64,device=DEVICE), tA, tZ).cpu().numpy())
            print(f"  ep{ep:4d}: ev={ev.item():+.3f} ec={ec.item():+.3f} AUROC={au:.3f}")
    with torch.no_grad():
        tZ = proj(t_feats.to(DEVICE)).reshape(200, h_dim, 2, 64)
        auc = roc_auc_score(tl.numpy(), energy(torch.zeros(200,2,64,device=DEVICE), tA, tZ).cpu().numpy())
    print(f"  Final: {auc:.3f}")
    if mode_name == "Frame diff":
        au_diff = auc
    else:
        au_raw = auc

print(f"\nFrame diff: {au_diff:.3f} | Raw frames: {au_raw:.3f} | Delta: {au_diff-au_raw:+.3f}")
