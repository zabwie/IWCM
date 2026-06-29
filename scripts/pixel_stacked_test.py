"""Stacked frames (2 consecutive) as 6-channel frozen ResNet input."""
import torch, torch.nn as nn, torch.nn.functional as F, numpy as np, sys
import torchvision.models as models
from sklearn.metrics import roc_auc_score
sys.path.insert(0, '.')
from src.env.dm_control_wrapper import DMControlWrapper
from src.iwcm.fused_energy import FusedIWCMEnergy

DEVICE = 'cuda'; H = 25
torch.manual_seed(42); np.random.seed(42); rng = np.random.RandomState(42)

# Build ResNet-18 up to layer3, replace conv1 for 6-channel input
rn18 = models.resnet18(weights='DEFAULT')
old_conv1 = rn18.conv1
new_conv1 = nn.Conv2d(6, 64, kernel_size=7, stride=2, padding=3, bias=False)
with torch.no_grad():
    new_conv1.weight[:, :3] = old_conv1.weight  # copy RGB for frame_t
    new_conv1.weight[:, 3:] = old_conv1.weight.mean(dim=1, keepdim=True)  # init frame_t+1
bb = nn.Sequential(new_conv1, rn18.bn1, rn18.relu, rn18.maxpool, rn18.layer1, rn18.layer2, rn18.layer3)
bb.eval().requires_grad_(False).to(DEVICE)
pool = nn.AdaptiveAvgPool2d(1)
proj = nn.Linear(256, 128).to(DEVICE)
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
    data, attempts, max_attempts = [], 0, n * 20
    while len(data) < n and attempts < max_attempts:
        r = gen(valid)
        attempts += 1
        if r is not None: data.append(r)
    return data

def stacked_feats(frames):
    B, H_ = frames.shape[:2]
    x = torch.from_numpy(frames.reshape(-1, 64, 64, 3).copy()).permute(0, 3, 1, 2).float() / 255.0
    pairs = []
    for b in range(B):
        for t in range(H_ - 1):
            idx = b * H_ + t
            pairs.append(torch.cat([x[idx:idx+1], x[idx+1:idx+2]], dim=1))
    x_stacked = torch.cat(pairs, dim=0).to(DEVICE)
    f = bb(x_stacked)
    return pool(f).reshape(B, H_-1, -1).cpu()

def raw_feats(frames):
    B, H_ = frames.shape[:2]
    x = torch.from_numpy(frames.reshape(-1, 64, 64, 3).copy()).permute(0, 3, 1, 2).float() / 255.0
    f = bb[:1](x.to(DEVICE))  # just conv1 — doesn't work with 3ch input to 6ch conv1
    return None

print("Generating data (500v+500c train, 100v+100c test)...")
import time; t0 = time.time()
tv = collect(500, True); tc = collect(500, False)
tev = collect(100, True); tec = collect(100, False)
print(f"  {time.time()-t0:.1f}s")

print("\n=== Stacked frames (6ch conv1) ===")
v_feats = stacked_feats(np.stack([d[0] for d in tv]))
c_feats = stacked_feats(np.stack([d[0] for d in tc]))
t_feats = stacked_feats(np.stack([d[0] for d in tev+tec]))
vA = torch.from_numpy(np.stack([d[1] for d in tv])[:, 1:]).float().to(DEVICE)
cA = torch.from_numpy(np.stack([d[1] for d in tc])[:, 1:]).float().to(DEVICE)
tA = torch.from_numpy(np.stack([d[1] for d in tev+tec])[:, 1:]).float().to(DEVICE)
tl = torch.cat([torch.zeros(100), torch.ones(100)])

for ep in range(301):
    vi=np.random.choice(500,64); ci=np.random.choice(500,64)
    zv = proj(v_feats[vi].to(DEVICE)).reshape(64, 24, 2, 64)
    zc = proj(c_feats[ci].to(DEVICE)).reshape(64, 24, 2, 64)
    Z = torch.cat([zv, zc]); A = torch.cat([vA[vi], cA[ci]])
    opt.zero_grad()
    e = energy(torch.zeros(128,2,64,device=DEVICE), A, Z)
    ev, ec = e[:64].mean(), e[64:].mean()
    loss = F.relu(ev+1) + F.relu(1-ec) + 1e-3*(ev**2+ec**2)
    loss.backward(); nn.utils.clip_grad_norm_(opt.param_groups[0]['params'],1.0); opt.step()
    if ep%100==0:
        with torch.no_grad():
            tZ = proj(t_feats.to(DEVICE)).reshape(200, 24, 2, 64)
            au = roc_auc_score(tl.numpy(), energy(torch.zeros(200,2,64,device=DEVICE), tA, tZ).cpu().numpy())
        print(f"  ep{ep:4d}: ev={ev.item():+.3f} ec={ec.item():+.3f} AUROC={au:.3f}")
with torch.no_grad():
    tZ = proj(t_feats.to(DEVICE)).reshape(200, 24, 2, 64)
    au = roc_auc_score(tl.numpy(), energy(torch.zeros(200,2,64,device=DEVICE), tA, tZ).cpu().numpy())
print(f"  Final: {au:.3f}")

# For fair comparison, run raw frames baseline with same 6ch conv1
# (use the first 3 channels only)
print("\n=== Raw frames (6ch conv1, zero-padded) ===")
proj2 = nn.Linear(256, 128).to(DEVICE)
energy2 = FusedIWCMEnergy(64, 1, 2).to(DEVICE)
opt2 = torch.optim.Adam(list(proj2.parameters())+list(energy2.parameters()), lr=1e-3)

def raw_6ch_feats(frames):
    B, H_ = frames.shape[:2]
    x = torch.from_numpy(frames.reshape(-1, 64, 64, 3).copy()).permute(0, 3, 1, 2).float()/255.0
    # Pad to 6 channels: [R,G,B,0,0,0]
    pad = torch.zeros(x.size(0), 3, 64, 64)
    x6 = torch.cat([x, pad], dim=1).to(DEVICE)
    f = bb(x6)
    return pool(f).reshape(B, H_, -1).cpu()

v_feats_r = raw_6ch_feats(np.stack([d[0] for d in tv]))
c_feats_r = raw_6ch_feats(np.stack([d[0] for d in tc]))
t_feats_r = raw_6ch_feats(np.stack([d[0] for d in tev+tec]))
vA_r = torch.from_numpy(np.stack([d[1] for d in tv])).float().to(DEVICE)
cA_r = torch.from_numpy(np.stack([d[1] for d in tc])).float().to(DEVICE)
tA_r = torch.from_numpy(np.stack([d[1] for d in tev+tec])).float().to(DEVICE)

for ep in range(301):
    vi=np.random.choice(500,64); ci=np.random.choice(500,64)
    zv = proj2(v_feats_r[vi].to(DEVICE)).reshape(64, 25, 2, 64)
    zc = proj2(c_feats_r[ci].to(DEVICE)).reshape(64, 25, 2, 64)
    Z = torch.cat([zv, zc]); A = torch.cat([vA_r[vi], cA_r[ci]])
    opt2.zero_grad()
    e = energy2(torch.zeros(128,2,64,device=DEVICE), A, Z)
    ev, ec = e[:64].mean(), e[64:].mean()
    loss = F.relu(ev+1) + F.relu(1-ec) + 1e-3*(ev**2+ec**2)
    loss.backward(); nn.utils.clip_grad_norm_(opt2.param_groups[0]['params'],1.0); opt2.step()
    if ep%100==0:
        with torch.no_grad():
            tZ = proj2(t_feats_r.to(DEVICE)).reshape(200, 25, 2, 64)
            au = roc_auc_score(tl.numpy(), energy2(torch.zeros(200,2,64,device=DEVICE), tA_r, tZ).cpu().numpy())
        print(f"  ep{ep:4d}: ev={ev.item():+.3f} ec={ec.item():+.3f} AUROC={au:.3f}")
with torch.no_grad():
    tZ = proj2(t_feats_r.to(DEVICE)).reshape(200, 25, 2, 64)
    au_r = roc_auc_score(tl.numpy(), energy2(torch.zeros(200,2,64,device=DEVICE), tA_r, tZ).cpu().numpy())
print(f"  Final: {au_r:.3f}")
