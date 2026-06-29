"""Test: does frozen backbone carry enough signal for IWCM?"""
import torch, torch.nn as nn, torch.nn.functional as F, numpy as np, sys, time
import torchvision.models as models
from sklearn.metrics import roc_auc_score
sys.path.insert(0, '.')
from src.env.dm_control_wrapper import DMControlWrapper
from src.iwcm.fused_energy import FusedIWCMEnergy

DEVICE = 'cuda'; H = 25; SEED = 42

rng = np.random.RandomState(SEED)
torch.manual_seed(SEED); np.random.seed(SEED)

wrapper = DMControlWrapper('cartpole', 'swingup')
env = wrapper._env

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

print("Generating data...")
t0 = time.time()
tv = collect(500, True); tc = collect(500, False)
tev = collect(100, True); tec = collect(100, False)
print(f"  {time.time()-t0:.1f}s: {len(tv)}v+{len(tc)}c train, {len(tev)}v+{len(tec)}c test")

# Try 5 approaches
approaches = []

# 1. Oracle baseline: train on physics state directly (not pixels)
print("\n1. Oracle slots (from physics state, not pixels)")
from src.env.dm_control_encoder import DMControlOracleEncoder
enc_oracle = DMControlOracleEncoder('cartpole')
def frames_to_oracle(frames, physics_states): pass  # skip, just for reference

# 2. Frozen ResNet layer4 → flatten → proj → 2×D slots
print("2. Frozen ResNet layer4 + full flatten + proj")
bb = nn.Sequential(*list(models.resnet18(weights='DEFAULT').children())[:-2])
bb.eval().requires_grad_(False).to(DEVICE)
proj = nn.Linear(2048, 128).to(DEVICE)
energy = FusedIWCMEnergy(64, 1, 2).to(DEVICE)
def enc(frames):
    x = torch.from_numpy(frames.reshape(-1,64,64,3).copy()).permute(0,3,1,2).float()/255.0
    f = bb(x.to(DEVICE))
    z = proj(f.reshape(f.size(0), -1))
    return z.reshape(frames.shape[0], frames.shape[1], 2, 64)
opt = torch.optim.Adam(list(proj.parameters())+list(energy.parameters()), lr=1e-3)
vz = enc(np.stack([d[0] for d in tv]))
vA = torch.stack([torch.from_numpy(d[1]) for d in tv]).float().to(DEVICE)
cz = enc(np.stack([d[0] for d in tc]))
cA = torch.stack([torch.from_numpy(d[1]) for d in tc]).float().to(DEVICE)
tz = enc(np.stack([d[0] for d in tev+tec]))
tA = torch.stack([torch.from_numpy(d[1]) for d in tev+tec]).float().to(DEVICE)
tl = torch.cat([torch.zeros(100), torch.ones(100)])
for ep in range(301):
    vi=np.random.choice(500,64); ci=np.random.choice(500,64)
    Z=torch.cat([vz[vi],cz[ci]]).to(DEVICE); A=torch.cat([vA[vi],cA[ci]])
    opt.zero_grad()
    e=energy(torch.zeros(128,2,64,device=DEVICE),A,Z)
    ev,ec=e[:64].mean(),e[64:].mean()
    loss=F.relu(ev+1)+F.relu(1-ec)+1e-3*(ev**2+ec**2)
    loss.backward(); nn.utils.clip_grad_norm_(opt.param_groups[0]['params'],1.0); opt.step()
    if ep%100==0:
        with torch.no_grad():
            au=roc_auc_score(tl.numpy(), energy(torch.zeros(200,2,64,device=DEVICE), tA, tz.to(DEVICE)).cpu().numpy())
        print(f"  ep{ep:4d}: ev={ev.item():+.3f} ec={ec.item():+.3f} AUROC={au:.3f}")
# Try with grad clipping removed
print(f"  Final: {roc_auc_score(tl.numpy(), energy(torch.zeros(200,2,64,device=DEVICE), tA, tz.to(DEVICE)).cpu().numpy()):.3f}")
" 2>&1