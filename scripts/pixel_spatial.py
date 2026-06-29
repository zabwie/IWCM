"""Test: spatial attention on native-resolution features (not GAP)."""
import torch, torch.nn as nn, torch.nn.functional as F, numpy as np, sys, math, time
import torchvision.models as models
from sklearn.metrics import roc_auc_score
sys.path.insert(0, '.')
from src.env.dm_control_wrapper import DMControlWrapper
from src.iwcm.fused_energy import FusedIWCMEnergy

DEVICE = 'cuda'; H = 25
torch.manual_seed(42); np.random.seed(42); rng = np.random.RandomState(42)

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
    data, attempts = [], 0
    while len(data) < n and attempts < n * 20:
        r = gen(valid); attempts += 1
        if r is not None: data.append(r)
    return data

print("Generating data...", end=" ", flush=True)
t0 = time.time()
tv = collect(500, True); tc = collect(500, False)
tev = collect(100, True); tec = collect(100, False)
print(f"{time.time()-t0:.1f}s")

backbones = {
    "layer1_16x16": {
        "bb": lambda: nn.Sequential(*list(models.resnet18(weights='DEFAULT').children())[:5]),
        "c": 64, "h": 16, "w": 16, "pos": 256,
    },
    "layer2_8x8": {
        "bb": lambda: nn.Sequential(*list(models.resnet18(weights='DEFAULT').children())[:6]),
        "c": 128, "h": 8, "w": 8, "pos": 64,
    },
    "layer3_4x4": {
        "bb": lambda: nn.Sequential(*list(models.resnet18(weights='DEFAULT').children())[:7]),
        "c": 256, "h": 4, "w": 4, "pos": 16,
    },
}

for name, cfg in backbones.items():
    print(f"\n=== {name} ({cfg['c']}ch, {cfg['h']}x{cfg['w']}) ===")
    
    bb = cfg["bb"]()
    bb.eval().requires_grad_(False).to(DEVICE)
    n_pos = cfg["pos"]  # H*W
    c, h, w = cfg["c"], cfg["h"], cfg["w"]
    
    # Lightweight slot attention: learned queries attend over spatial positions
    # pos_embed: (1, n_pos, c) — helps encode where each feature is
    # slot_queries: (1, 2, d_slot) — 2 learnable slot queries
    # proj_in: Linear(c, 128) — project spatial features to attention dim
    # proj_out: Linear(128, 2*64) — project attended features to 2×64 slots
    
    proj_in = nn.Linear(c, 128).to(DEVICE)
    proj_out = nn.Linear(128, 64).to(DEVICE)
    pos_embed = nn.Parameter(torch.randn(1, n_pos, 128, device=DEVICE) * 0.02)
    slot_queries = nn.Parameter(torch.randn(1, 2, 128, device=DEVICE) * 0.02)
    energy = FusedIWCMEnergy(64, 1, 2).to(DEVICE)
    params = list(proj_in.parameters()) + list(proj_out.parameters()) + [pos_embed, slot_queries] + list(energy.parameters())
    opt = torch.optim.Adam(params, lr=1e-3)
    
    def raw_feats(frames_batch):
        B = frames_batch.shape[0]
        x = torch.from_numpy(frames_batch.reshape(-1, 64, 64, 3).copy()).permute(0, 3, 1, 2).float() / 255.0
        all_feats = []
        for i in range(0, B, 32):
            chunk = x[i*H:(i+32)*H].to(DEVICE) if (i+32)*H <= len(x) else x[i*H:].to(DEVICE)
            f = bb(chunk)
            FH = f.shape[0]
            all_feats.append(f.reshape(FH, c, n_pos).permute(0, 2, 1).cpu())
        return torch.cat(all_feats, dim=0).reshape(B, H, n_pos, c)
    
    def attn_to_slots(raw_feats_t):
        """raw_feats_t: (B, H, n_pos, C) on DEVICE → (B, H, 2, 64)"""
        B, H_, n_pos_, C_ = raw_feats_t.shape
        f_flat = raw_feats_t.reshape(B*H_, n_pos_, C_)
        f_proj = proj_in(f_flat) + pos_embed
        attn = F.softmax(torch.matmul(slot_queries, f_proj.transpose(1,2)) / math.sqrt(128), dim=-1)
        slots_raw = torch.matmul(attn, f_proj)
        return proj_out(slots_raw).reshape(B, H_, 2, 64)
    
    frames_v = np.stack([d[0] for d in tv])
    frames_c = np.stack([d[0] for d in tc])
    frames_te = np.stack([d[0] for d in tev+tec])
    
    v_raw = raw_feats(frames_v)
    c_raw = raw_feats(frames_c)
    t_raw = raw_feats(frames_te)
    
    vA = torch.stack([torch.from_numpy(d[1]) for d in tv]).float().to(DEVICE)
    cA = torch.stack([torch.from_numpy(d[1]) for d in tc]).float().to(DEVICE)
    tA = torch.stack([torch.from_numpy(d[1]) for d in tev+tec]).float().to(DEVICE)
    tl = torch.cat([torch.zeros(100), torch.ones(100)])
    
    for ep in range(301):
        vi=np.random.choice(500,64); ci=np.random.choice(500,64)
        raw = torch.cat([v_raw[vi], c_raw[ci]]).to(DEVICE)
        Z = attn_to_slots(raw)
        A = torch.cat([vA[vi], cA[ci]])
        opt.zero_grad()
        e=energy(torch.zeros(128,2,64,device=DEVICE), A, Z)
        ev,ec=e[:64].mean(),e[64:].mean()
        loss=F.relu(ev+1)+F.relu(1-ec)+1e-3*(ev**2+ec**2)
        loss.backward(); nn.utils.clip_grad_norm_(params, 1.0); opt.step()
        if ep%100==0:
            with torch.no_grad():
                tZ = attn_to_slots(t_raw.to(DEVICE))
                au=roc_auc_score(tl.numpy(), energy(torch.zeros(200,2,64,device=DEVICE), tA, tZ).cpu().numpy())
            print(f"  ep{ep:4d}: ev={ev.item():+.3f} ec={ec.item():+.3f} AUROC={au:.3f}")
    with torch.no_grad():
        tZ = attn_to_slots(t_raw.to(DEVICE))
        au = roc_auc_score(tl.numpy(), energy(torch.zeros(200,2,64,device=DEVICE), tA, tZ).cpu().numpy())
    print(f"  Final: {au:.3f}")
