#!/usr/bin/env python3
"""TAMG-v2 from pixels — minimal self-contained pipeline.

1. Generate grid-world valid trajectories + render video frames
2. Generate oracle-invalid test trajectories (state-level corruptions) + render
3. Train CNN+slot encoder on valid video → learned latent Z
4. Self-supervised teacher on learned-Z pseudo-negatives
5. TAMG-v2 adversarial mining on learned Z
6. Evaluate IWCM on oracle-invalid test (oracle slots)

Target: TAMG-v2 pixels > 0.75 AUROC → viable. > 0.85 → strong.
"""
import sys, pickle, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from pathlib import Path
from copy import deepcopy
from collections import defaultdict
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
GRID = 8; CELL = 8; H_ = 12; N_SLOTS = 6; D_SLOT = 32
D_ACTION = 11; BATCH = 32; N_TRAIN = 300; N_TEST = 100
SEED = 42

def sseed(s):
    np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(s)

# ═══════════════════════════════════════════════════════════════════════════════
# 1. Data generation: grid world trajectories + video frames
# ═══════════════════════════════════════════════════════════════════════════════

from src.env.grid_world import GridWorld
from src.env.renderer import GridWorldRenderer
from src.env.oracle_slot_encoder import encode_oracle_trajectory, build_door_key_map

RENDERER = GridWorldRenderer(grid_size=GRID, cell_px=CELL)
FRAME_H, FRAME_W = RENDERER.img_size, RENDERER.img_size

SCENARIOS = [
    {"agent_start": (6,3), "goal": {"type":"position","pos":(0,3)},
     "objects": [{"type":"key","id":"k0","pos":(5,3)}, {"type":"door","id":"d0","pos":(2,3),"key_id":"k0"}]},
    {"agent_start": (4,1), "goal": {"type":"position","pos":(4,6)},
     "objects": [{"type":"box","id":"b0","pos":(4,3)}, {"type":"box","id":"b1","pos":(4,4)}]},
    {"agent_start": (7,1), "goal": {"type":"position","pos":(0,5)},
     "objects": [{"type":"key","id":"k0","pos":(6,5)}, {"type":"door","id":"d0","pos":(4,5),"key_id":"k0"},
                 {"type":"box","id":"b0","pos":(3,3)}]},
    {"agent_start": (2,2), "goal": {"type":"position","pos":(6,6)},
     "objects": [{"type":"occluder","id":"o0","pos":(3,4)}, {"type":"key","id":"k0","pos":(5,5)},
                 {"type":"door","id":"d0","pos":(5,2),"key_id":"k0"}]},
    {"agent_start": (1,5), "goal": {"type":"position","pos":(7,2)},
     "objects": [{"type":"box","id":"b0","pos":(3,5)}, {"type":"box","id":"b1","pos":(2,1)},
                 {"type":"key","id":"k0","pos":(0,6)}, {"type":"door","id":"d0","pos":(6,3),"key_id":"k0"}]},
]

def gen_trajectory(scenario, horizon, rng):
    gw = GridWorld(grid_size=GRID, objects_config=scenario, seed=int(rng.randint(0, 2**31)))
    gw.reset()
    states, actions, dk_map = [], [], build_door_key_map(scenario)
    states.append(deepcopy(gw.get_state()))
    for _ in range(horizon * 4):
        acts = gw.get_valid_actions()
        if not acts: break
        a = int(rng.choice(acts))
        st, _, done, _ = gw.step(a)
        states.append(deepcopy(st)); actions.append(a)
        if done: break
    if len(states) >= horizon + 1 and len(actions) >= horizon:
        return states, actions, dk_map
    return None

def state_to_oracle(states, actions, horizon, dk_map):
    return encode_oracle_trajectory(states, actions, horizon, GRID, None, dk_map)

def states_to_video(states, horizon):
    frames = [RENDERER.render_frame(states[i]) for i in range(horizon)]
    return np.stack(frames, axis=0)

def states_to_targets(states, horizon):
    positions = np.zeros((horizon, N_SLOTS, 2), dtype=np.float32)
    existence = np.zeros((horizon, N_SLOTS), dtype=np.float32)
    for t in range(horizon):
        st = states[t]
        ar, ac = st["agent_pos"]
        positions[t, 0] = [ar / GRID, ac / GRID]
        existence[t, 0] = 1.0
        slot = 1
        for oid, obj in st.get("objects", {}).items():
            if slot >= N_SLOTS: break
            r, c = obj["pos"]
            positions[t, slot] = [r / GRID, c / GRID]
            existence[t, slot] = 1.0
            slot += 1
    return torch.from_numpy(positions), torch.from_numpy(existence)
    frames = [RENDERER.render_frame(states[i]) for i in range(horizon)]
    return np.stack(frames, axis=0)  # (H, H_img, W_img, 3)

def corrupt_trajectory(states, actions, violation_type, rng):
    n = len(states)
    if n < H_ + 2: return None
    s = deepcopy(states)
    t = rng.randint(2, min(H_, n - 2))
    obj_keys = list(s[t].get("objects", {}).keys())
    if not obj_keys: return None

    if violation_type == "delete" and len(obj_keys) >= 1:
        k = obj_keys[rng.randint(0, len(obj_keys) - 1)]
        for i in range(t + 1, min(t + 1 + H_, n)):
            if k in s[i].get("objects", {}): del s[i]["objects"][k]
    elif violation_type == "duplicate" and len(obj_keys) >= 1:
        k = obj_keys[rng.randint(0, len(obj_keys) - 1)]
        if k in s[t].get("objects", {}):
            new_id = k + "_dup"
            for i in range(t, min(t + H_, n)):
                if k in s[i].get("objects", {}):
                    s[i]["objects"][new_id] = deepcopy(s[i]["objects"][k])
    elif violation_type == "swap" and len(obj_keys) >= 2:
        k1, k2 = obj_keys[:2]
        for i in range(t, min(t + H_, n)):
            if k1 in s[i].get("objects", {}) and k2 in s[i].get("objects", {}):
                s[i]["objects"][k1]["pos"], s[i]["objects"][k2]["pos"] = \
                    s[i]["objects"][k2]["pos"], s[i]["objects"][k1]["pos"]
    elif violation_type == "teleport":
        for k in obj_keys:
            if k in s[t].get("objects", {}):
                old_pos = s[t]["objects"][k]["pos"]
                new_pos = ((old_pos[0] + rng.randint(-4, 4)) % GRID,
                           (old_pos[1] + rng.randint(-4, 4)) % GRID)
                for i in range(t + 1, min(t + 1 + H_, n)):
                    if k in s[i].get("objects", {}):
                        s[i]["objects"][k]["pos"] = new_pos
                break
    elif violation_type == "transform":
        for k in obj_keys:
            if k in s[t].get("objects", {}):
                old_type = s[t]["objects"][k].get("type","key")
                new_type = "box" if old_type == "key" else "key"
                for i in range(t, min(t + H_, n)):
                    if k in s[i].get("objects", {}):
                        s[i]["objects"][k]["type"] = new_type
                break
    elif violation_type == "reverse":
        for i in range(t, min(t + H_ - 1, n - 1)):
            if i + 1 < len(actions):
                pass
    return s, actions, violation_type

print("Generating trajectories...")
rng = np.random.RandomState(SEED)

train_states_all, test_states_all = [], []
train_corr_all, test_corr_all = [], []

for _ in range(N_TRAIN + N_TEST):
    sc = SCENARIOS[rng.randint(0, len(SCENARIOS) - 1)]
    result = gen_trajectory(sc, H_, rng)
    if result is None: continue
    states, actions, dk_map = result
    if len(train_states_all) < N_TRAIN:
        train_states_all.append((states, actions, dk_map))
    elif len(test_states_all) < N_TEST:
        test_states_all.append((states, actions, dk_map))
    else:
        break

# Generate corrupted test trajectories
for states, actions, dk_map in test_states_all:
    vt = ["delete","duplicate","swap","teleport","transform"][rng.randint(0,4)]
    cr = corrupt_trajectory(states, actions, vt, rng)
    if cr is not None:
        test_corr_all.append(cr)

print(f"Train valid: {len(train_states_all)}, Test valid: {len(test_states_all)}, Test corrupt: {len(test_corr_all)}")

# Render corrupted test videos too, encode through learned encoder for evaluation
print("Rendering corrupted test videos...")
test_corr_videos = []
test_corr_labels = []
for states, actions, vt in test_corr_all:
    if len(states) >= H_ + 1:
        vid = states_to_video(states, H_)
        test_corr_videos.append(torch.from_numpy(vid).float().permute(0, 3, 1, 2) / 255.0)
        test_corr_labels.append(vt)

test_corr_videos_t = torch.stack(test_corr_videos)
print(f"Corrupt test videos: {test_corr_videos_t.shape}")

# Also compute oracle-slot test data for AUROC evaluation
test_os_valid = []
for states, actions, dk_map in test_states_all:
    enc = state_to_oracle(states, actions, H_, dk_map)
    if enc is not None: test_os_valid.append(enc)

test_os_corr = []
test_os_corr_vt = []
for states, actions, vt in test_corr_all:
    enc = state_to_oracle(states, actions, H_, build_door_key_map({}))
    if enc is not None:
        test_os_corr.append(enc)
        test_os_corr_vt.append(vt)

VTYPES = sorted(set(test_os_corr_vt))
test_os_by_type = defaultdict(list)
for (z0, A, Z), vt in zip(test_os_corr, test_os_corr_vt):
    test_os_by_type[vt].append((torch.from_numpy(z0).float(),
                                 torch.from_numpy(A).float(),
                                 torch.from_numpy(Z).float()))

print(f"Oracle-slot test: {len(test_os_valid)}v + {len(test_os_corr)}c, types: {VTYPES}")
for vt in VTYPES: print(f"  {vt}: {len(test_os_by_type[vt])}")

# Render video frames for training
print("Rendering video frames...")
train_videos = []
for states, actions, dk_map in train_states_all:
    vid = states_to_video(states, H_)
    train_videos.append(torch.from_numpy(vid).float().permute(0, 3, 1, 2) / 255.0)

test_videos = []
for states, actions, dk_map in test_states_all:
    vid = states_to_video(states, H_)
    test_videos.append(torch.from_numpy(vid).float().permute(0, 3, 1, 2) / 255.0)

train_videos_t = torch.stack(train_videos)
test_videos_t = torch.stack(test_videos)

train_targets_pos = []
train_targets_exist = []
for states, actions, dk_map in train_states_all:
    p, e = states_to_targets(states, H_)
    train_targets_pos.append(p)
    train_targets_exist.append(e)
train_pos_t = torch.stack(train_targets_pos)
train_exist_t = torch.stack(train_targets_exist)

print(f"Videos: train={train_videos_t.shape}, test={test_videos_t.shape}")
print(f"Oracle test: {len(test_os_valid)}v + {len(test_os_corr)}c, types: {VTYPES}")
for vt in VTYPES: print(f"  {vt}: {len(test_os_by_type[vt])}")

# ═══════════════════════════════════════════════════════════════════════════════
# 2. CNN + Slot Attention encoder
# ═══════════════════════════════════════════════════════════════════════════════

class SlotAttention(nn.Module):
    def __init__(self, num_slots, slot_dim, input_dim, iters=3):
        super().__init__()
        self.num_slots = num_slots; self.slot_dim = slot_dim; self.iters = iters
        self.slots_mu = nn.Parameter(torch.randn(1, 1, num_slots, slot_dim) * 0.02)
        self.slots_logsigma = nn.Parameter(torch.zeros(1, 1, num_slots, slot_dim))
        self.q = nn.Linear(slot_dim, slot_dim, bias=False)
        self.k = nn.Linear(input_dim, slot_dim, bias=False)
        self.v = nn.Linear(input_dim, slot_dim, bias=False)
        self.gru = nn.GRUCell(slot_dim, slot_dim)
        self.mlp = nn.Sequential(
            nn.Linear(slot_dim, slot_dim * 2), nn.ReLU(), nn.Linear(slot_dim * 2, slot_dim))

    def forward(self, inputs):
        B, N, d_in = inputs.shape
        slots = self.slots_mu + torch.randn(B, N, self.num_slots, self.slot_dim,
                                             device=inputs.device) * self.slots_logsigma.exp().sqrt()
        slots = slots.reshape(-1, self.slot_dim)
        inputs_flat = inputs.reshape(B * N, d_in)
        k = self.k(inputs_flat); v = self.v(inputs_flat)
        for _ in range(self.iters):
            q = self.q(slots)
            attn = F.softmax((q @ k.T) / (self.slot_dim ** 0.5), dim=-1)
            updates = attn @ v
            slots = self.gru(updates, slots)
            slots = slots + self.mlp(slots)
        slots = slots.reshape(B, self.num_slots, self.slot_dim)
        attn = attn.reshape(B * N, self.num_slots, -1).mean(dim=0)
        return slots

class SimpleVideoEncoder(nn.Module):
    def __init__(self, slot_dim=D_SLOT, num_slots=N_SLOTS):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 16, 4, 2, 1), nn.ReLU(),
            nn.Conv2d(16, 32, 4, 2, 1), nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2, 1), nn.ReLU(),
        )
        self.proj = nn.Linear(64 * 8 * 8, 64)
        self.slot_attn = SlotAttention(num_slots=num_slots, slot_dim=slot_dim, input_dim=64)
        self.pos_embed = nn.Parameter(torch.zeros(1, H_, 1, slot_dim))
        self.pos_head = nn.Linear(slot_dim, 2)
        self.exist_head = nn.Sequential(nn.Linear(slot_dim, 32), nn.ReLU(), nn.Linear(32, 1))
        self.temporal = nn.GRU(slot_dim, slot_dim, batch_first=True)

    def forward(self, video, return_aux=False):
        B, H_, C, H_img, W_img = video.shape
        frames = video.reshape(B * H_, C, H_img, W_img)
        feats = self.cnn(frames)
        feats = feats.reshape(B * H_, -1)
        feats = self.proj(feats).reshape(B, H_, -1)
        Z_raw = self.slot_attn(feats.reshape(-1, 1, 64)).reshape(B, H_, N_SLOTS, D_SLOT)
        Z = Z_raw + self.pos_embed
        Z_t, _ = self.temporal(Z.reshape(B * N_SLOTS, H_, D_SLOT))
        Z = Z_t.reshape(B, H_, N_SLOTS, D_SLOT) + Z_raw * 0.5
        if return_aux:
            pos = torch.tanh(self.pos_head(Z))
            exist = torch.sigmoid(self.exist_head(Z)).squeeze(-1)
            return Z, pos, exist
        return Z

class SimpleDecoder(nn.Module):
    def __init__(self, slot_dim=D_SLOT):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(slot_dim, 64 * 8 * 8), nn.ReLU())
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(16, 3, 4, 2, 1), nn.Sigmoid(),
        )

    def forward(self, Z):
        B, H_, Ns, d = Z.shape
        Zp = Z.mean(dim=2).reshape(B * H_, d)
        feats = self.fc(Zp).reshape(B * H_, 64, 8, 8)
        recon = self.deconv(feats)
        return recon.reshape(B, H_, 3, 64, 64)

print("\nTraining encoder (reconstruction + position + existence)...")
encoder = SimpleVideoEncoder().to(DEVICE)
decoder = SimpleDecoder().to(DEVICE)
opt_enc = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=1e-3)

for ep in range(400):
    idx = np.random.choice(len(train_videos_t), min(BATCH, len(train_videos_t)), replace=False)
    vid = train_videos_t[idx].to(DEVICE)
    Z, pos_pred, exist_pred = encoder(vid, return_aux=True)
    recon = decoder(Z)
    loss = F.mse_loss(recon, vid)
    pos_target = train_pos_t[idx].to(DEVICE)
    exist_target = train_exist_t[idx].to(DEVICE)
    loss = loss + 0.1 * F.mse_loss(pos_pred, pos_target)
    loss = loss + 0.5 * F.binary_cross_entropy(exist_pred, exist_target)
    opt_enc.zero_grad(); loss.backward(); opt_enc.step()
    if (ep + 1) % 100 == 0:
        exist_acc = ((exist_pred > 0.5).float() == exist_target).float().mean()
        print(f"  ep{ep+1:3d}: recon={F.mse_loss(recon,vid).item():.4f}  "
              f"pos_mse={F.mse_loss(pos_pred,pos_target).item():.4f}  "
              f"exist_acc={exist_acc.item():.3f}")

# ═══════════════════════════════════════════════════════════════════════════════
# 3. Encode train/test data to learned Z
# ═══════════════════════════════════════════════════════════════════════════════

encoder.eval()
decoder.eval()

@torch.no_grad()
def encode_videos(videos):
    Zs = []
    for i in range(0, len(videos), BATCH):
        batch = videos[i:i+BATCH].to(DEVICE)
        Zs.append(encoder(batch).cpu())
    return torch.cat(Zs, dim=0)

print("Encoding videos to learned Z...")
train_Z = encode_videos(train_videos_t)
test_Z_valid = encode_videos(test_videos_t)
test_Z_corr = encode_videos(test_corr_videos_t)

train_valid_learned = []
for i in range(len(train_Z)):
    z0 = train_Z[i, 0].mean(dim=0)
    A = torch.zeros(H_, D_ACTION)
    train_valid_learned.append((z0, A, train_Z[i]))

test_valid_learned = []
for i in range(len(test_Z_valid)):
    z0 = test_Z_valid[i, 0].mean(dim=0)
    A = torch.zeros(H_, D_ACTION)
    test_valid_learned.append((z0, A, test_Z_valid[i]))

test_corr_learned = []
test_corr_learned_vt = []
for i in range(len(test_Z_corr)):
    z0 = test_Z_corr[i, 0].mean(dim=0)
    A = torch.zeros(H_, D_ACTION)
    test_corr_learned.append((z0, A, test_Z_corr[i]))
    test_corr_learned_vt.append(test_corr_labels[i])

test_lz_by_type = defaultdict(list)
for idx, vt in enumerate(test_corr_learned_vt):
    test_lz_by_type[vt].append(test_corr_learned[idx])

print(f"Learned-Z test: {len(test_valid_learned)}v + {len(test_corr_learned)}c")
for vt in sorted(set(test_corr_learned_vt)):
    print(f"  {vt}: {len(test_lz_by_type[vt])}")

class SlotPoolTeacher(nn.Module):
    def __init__(self):
        super().__init__()
        self.slot_encoder = nn.Sequential(nn.Linear(D_SLOT * 3, 128), nn.ReLU())
        self.cross_slot = nn.Sequential(nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.1), nn.Linear(128, 1))
    def forward(self, Z):
        B, H_, Ns, d = Z.shape
        m = Z.mean(dim=1); mx = Z.amax(dim=1)
        v = F.relu((Z * Z).mean(dim=1) - m * m)
        Zp = torch.cat([m, mx, torch.sqrt(v + 1e-5)], dim=-1)
        sf = self.slot_encoder(Zp)
        return self.cross_slot(torch.cat([sf.mean(dim=1), sf.amax(dim=1)], dim=-1)).squeeze(-1)

def passes_filter(Zo, Zc):
    p = (Zc - Zo).pow(2).mean().sqrt().item()
    if p > 2.0: return False
    cn = Zc.norm(dim=-1)
    ac = (cn > 0.1).float().sum(dim=-1).float().mean().item()
    if ac < 0.5: return False
    return True

def gen_mixed_learned(data):
    corr = []
    for z0, A, Z in data:
        for _ in range(2):
            Zc = Z.clone()
            fam = np.random.randint(0, 6)
            if fam == 0:
                t = np.random.randint(1, H_); Zc[t:] = Zc[:H_-t].clone()
            elif fam == 1 and N_SLOTS >= 2:
                i, j = np.random.choice(N_SLOTS, 2, replace=False)
                Zc[:, i], Zc[:, j] = Zc[:, j].clone(), Zc[:, i].clone()
            elif fam == 2:
                Zc[:, np.random.randint(0, N_SLOTS)] *= 0.0
            elif fam == 3 and N_SLOTS >= 2:
                i, j = np.random.choice(N_SLOTS, 2, replace=False)
                Zc[:, j] = Zc[:, i].clone()
            elif fam == 4:
                Zc += torch.randn_like(Zc) * 0.03
            elif fam == 5:
                slot = np.random.randint(0, N_SLOTS)
                t = np.random.randint(1, H_ - 1)
                Zc[t:, slot] += (torch.rand(D_SLOT) * 2 - 1) * 0.1
            if passes_filter(Z, Zc):
                corr.append((z0.clone(), A.clone(), Zc))
    return corr

print("\nTraining teacher on learned-Z pseudo-negatives...")
mixed_corr_learned = gen_mixed_learned(train_valid_learned)
sseed(SEED)
teacher = SlotPoolTeacher().to(DEVICE)
opt_t = torch.optim.Adam(teacher.parameters(), lr=1e-3, weight_decay=1e-5)
for ep in range(150):
    vi = np.random.choice(len(train_valid_learned), min(BATCH, len(train_valid_learned)), replace=False)
    ci = np.random.choice(len(mixed_corr_learned), min(BATCH, len(mixed_corr_learned)), replace=False)
    Zv = torch.stack([train_valid_learned[i][2] for i in vi]).to(DEVICE)
    Zc = torch.stack([mixed_corr_learned[i][2] for i in ci]).to(DEVICE)
    Zb = torch.cat([Zv, Zc], dim=0)
    y = torch.cat([torch.zeros(len(vi)), torch.ones(len(ci))]).to(DEVICE)
    opt_t.zero_grad()
    loss = F.binary_cross_entropy_with_logits(teacher(Zb), y)
    loss.backward(); opt_t.step()
teacher.eval()
for p in teacher.parameters(): p.requires_grad = False

with torch.no_grad():
    tv_s = [torch.sigmoid(teacher(test_valid_learned[i][2].unsqueeze(0).to(DEVICE))).item()
            for i in range(len(test_valid_learned))]
    tc_s = [torch.sigmoid(teacher(test_corr_learned[i][2].unsqueeze(0).to(DEVICE))).item()
            for i in range(len(test_corr_learned))]
    t_auroc_lz = roc_auc_score([0]*len(tv_s)+[1]*len(tc_s), tv_s+tc_s)
    print(f"SlotPool teacher AUROC: {t_auroc_lz:.4f}")

    tv_pos = encode_videos(test_videos_t)
    tc_pos = encode_videos(test_corr_videos_t)
    _, pos_v, exist_v = encoder(test_videos_t.to(DEVICE), return_aux=True)
    _, pos_c, exist_c = encoder(test_corr_videos_t.to(DEVICE), return_aux=True)
    pos_v = pos_v.cpu(); pos_c = pos_c.cpu(); exist_v = exist_v.cpu(); exist_c = exist_c.cpu()

    pos_diff_valid = (pos_v[:, 1:] - pos_v[:, :-1]).abs().amax(dim=(-2, -1)).mean(dim=1).numpy()
    pos_diff_corr = (pos_c[:, 1:] - pos_c[:, :-1]).abs().amax(dim=(-2, -1)).mean(dim=1).numpy()
    pos_auroc = roc_auc_score([0]*len(pos_diff_valid)+[1]*len(pos_diff_corr),
                               np.concatenate([pos_diff_valid, pos_diff_corr]))

    exist_min_valid = exist_v.amin(dim=(-2, -1)).numpy()
    exist_min_corr = exist_c.amin(dim=(-2, -1)).numpy()
    exist_auroc = roc_auc_score([0]*len(exist_min_valid)+[1]*len(exist_min_corr),
                                 np.concatenate([exist_min_valid, exist_min_corr]))

    combined = (1 - np.concatenate([exist_min_valid, exist_min_corr])) + \
               np.concatenate([pos_diff_valid, pos_diff_corr])
    comb_auroc = roc_auc_score([0]*len(pos_diff_valid)+[1]*len(pos_diff_corr), combined)
    print(f"Aux-head validator: position={pos_auroc:.4f}  existence={exist_auroc:.4f}  combined={comb_auroc:.4f}")

# ---------- TAMG-v2 adversarial mining on learned Z ----------
print("\nTAMG-v2 adversarial mining (learned Z)...")

def train_tamg_v2_learned(seed):
    sseed(seed)
    from src.iwcm.fused_energy import FusedIWCMEnergy
    model = FusedIWCMEnergy(d_slot=D_SLOT, d_action=D_ACTION, hidden=128,
                            num_slots=N_SLOTS).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    nv = len(train_valid_learned)
    for ep in range(200):
        corr = gen_mixed_learned(train_valid_learned)
        if len(corr) < BATCH: continue
        pool_Z = torch.stack([c[2] for c in corr]).to(DEVICE)
        pool_z0 = torch.stack([c[0] for c in corr]).to(DEVICE)
        pool_A  = torch.stack([c[1] for c in corr]).to(DEVICE)
        with torch.no_grad():
            t_scores = torch.sigmoid(teacher(pool_Z)).cpu().numpy()
            energies = model(pool_z0, pool_A, pool_Z).cpu().numpy()
        invalid_mask = t_scores > 0.3
        low_e_mask = energies < 1.5
        good_idx = np.where(invalid_mask & low_e_mask)[0]
        if len(good_idx) < BATCH // 2: good_idx = np.where(invalid_mask)[0]
        if len(good_idx) < BATCH // 2: good_idx = np.arange(len(corr))
        chosen = np.random.choice(good_idx, min(BATCH, len(good_idx)), replace=False)
        cZ_b = pool_Z[chosen].clone().detach().requires_grad_(True)
        cz0_b = pool_z0[chosen]; cA_b = pool_A[chosen]
        adv_opt = torch.optim.Adam([cZ_b], lr=0.01)
        for _ in range(3):
            adv_opt.zero_grad()
            t_a = torch.sigmoid(teacher(cZ_b))
            e_a = model(cz0_b, cA_b, cZ_b)
            adv_loss = -(t_a.mean() - 0.3 * e_a.mean())
            adv_loss.backward(); adv_opt.step()
            cZ_b.data = cZ_b.data.clamp(-5.0, 5.0)
        cZ_f = cZ_b.detach()
        vi = np.random.choice(nv, min(BATCH, nv), replace=False)
        vz0 = torch.stack([train_valid_learned[i][0] for i in vi]).to(DEVICE)
        vA  = torch.stack([train_valid_learned[i][1] for i in vi]).to(DEVICE)
        vZ  = torch.stack([train_valid_learned[i][2] for i in vi]).to(DEVICE)
        with torch.no_grad():
            tw = torch.sigmoid(teacher(cZ_f)).clamp(0.1, 2.0)
        opt.zero_grad()
        ev = model(vz0, vA, vZ); ec = model(cz0_b, cA_b, cZ_f)
        loss = F.relu(ev + 1.0).mean() + (tw * F.relu(1.0 - ec)).mean() + \
               0.001 * (ev.pow(2).mean() + ec.pow(2).mean())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
    return model.eval()

# ---------- Evaluate on learned-Z test ----------
print("\nEvaluating TAMG-v2 on learned-Z test...")
results = {}
for seed in [42, 123, 456]:
    torch.cuda.empty_cache()
    model = train_tamg_v2_learned(seed)
    with torch.no_grad():
        ve = [model(z0.unsqueeze(0).to(DEVICE), A.unsqueeze(0).to(DEVICE),
                    Z.unsqueeze(0).to(DEVICE)).item() for z0, A, Z in test_valid_learned]
        per_type = {}
        for vt in sorted(test_lz_by_type.keys()):
            ce = [model(z0.unsqueeze(0).to(DEVICE), A.unsqueeze(0).to(DEVICE),
                        Z.unsqueeze(0).to(DEVICE)).item() for z0, A, Z in test_lz_by_type[vt]]
            per_type[vt] = roc_auc_score([0]*len(ve)+[1]*len(ce), ve+ce)
        all_ce = [model(z0.unsqueeze(0).to(DEVICE), A.unsqueeze(0).to(DEVICE),
                        Z.unsqueeze(0).to(DEVICE)).item() for z0, A, Z in test_corr_learned]
        per_type["OVERALL"] = roc_auc_score(
            [0]*len(ve)+[1]*len(all_ce), ve+all_ce)
    results[seed] = per_type
    print(f"  seed={seed}: OVERALL={per_type['OVERALL']:.4f}  " +
          " ".join(f"{vt[:3]}={per_type[vt]:.3f}" for vt in sorted(test_lz_by_type.keys())[:4]))

# ---------- Oracle-slot IWCM baseline ----------
print("\nOracle-slot IWCM baseline...")
def train_iwcm_oracle(seed):
    sseed(seed)
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from src.iwcm.fused_energy import FusedIWCMEnergy as FE
    model = FE(d_slot=19, d_action=D_ACTION, hidden=128, num_slots=8).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    tv_os = [(torch.from_numpy(z0).float(), torch.from_numpy(A).float(),
              torch.from_numpy(Z).float()) for z0, A, Z in test_os_valid]
    tc_os = [(torch.from_numpy(z0).float(), torch.from_numpy(A).float(),
              torch.from_numpy(Z).float()) for z0, A, Z in test_os_corr]
    nv, nc = len(tv_os), len(tc_os)
    for ep in range(150):
        vi = np.random.choice(nv, min(BATCH, nv), replace=False)
        ci = np.random.choice(nc, min(BATCH, nc), replace=False)
        vz0 = torch.stack([tv_os[i][0] for i in vi]).to(DEVICE)
        vA  = torch.stack([tv_os[i][1] for i in vi]).to(DEVICE)
        vZ  = torch.stack([tv_os[i][2] for i in vi]).to(DEVICE)
        cz0 = torch.stack([tc_os[i][0] for i in ci]).to(DEVICE)
        cA  = torch.stack([tc_os[i][1] for i in ci]).to(DEVICE)
        cZ  = torch.stack([tc_os[i][2] for i in ci]).to(DEVICE)
        opt.zero_grad()
        ev = model(vz0, vA, vZ); ec = model(cz0, cA, cZ)
        loss = F.relu(ev+1.0).mean() + F.relu(1.0-ec).mean() + \
               0.001*(ev.pow(2).mean()+ec.pow(2).mean())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
    model.eval()
    with torch.no_grad():
        ve = [model(z0.unsqueeze(0).to(DEVICE), A.unsqueeze(0).to(DEVICE),
                    Z.unsqueeze(0).to(DEVICE)).item() for z0, A, Z in tv_os]
        per_type = {}
        for vt in VTYPES:
            ce = [model(z0.unsqueeze(0).to(DEVICE), A.unsqueeze(0).to(DEVICE),
                        Z.unsqueeze(0).to(DEVICE)).item() for z0, A, Z in test_os_by_type[vt]]
            per_type[vt] = roc_auc_score([0]*len(ve)+[1]*len(ce), ve+ce)
        all_ce = []
        for vt in VTYPES:
            for z0, A, Z in test_os_by_type[vt]:
                all_ce.append(model(z0.unsqueeze(0).to(DEVICE), A.unsqueeze(0).to(DEVICE),
                                   Z.unsqueeze(0).to(DEVICE)).item())
        per_type["OVERALL"] = roc_auc_score([0]*len(ve)+[1]*len(all_ce), ve+all_ce)
    return per_type

os_results = {}
for seed in [42, 123, 456]:
    torch.cuda.empty_cache()
    os_results[seed] = train_iwcm_oracle(seed)
    print(f"  seed={seed}: OVERALL={os_results[seed]['OVERALL']:.4f}")

# ---------- Verdict ----------
print("\n" + "=" * 65)
print("PIXEL TAMG-v2 VERDICT")
print("=" * 65)

print(f"\n{'Type':<14} {'TAMG-v2 (LZ)':>14} {'Oracle-slot':>14}")
print("-" * 46)
lz_vtypes = sorted(test_lz_by_type.keys())
for vt in lz_vtypes:
    lz_mean = np.mean([results[s][vt] for s in [42, 123, 456]])
    os_mean = np.mean([os_results[s].get(vt, 0) for s in [42, 123, 456]])
    print(f"{vt:<14} {lz_mean:14.4f} {os_mean:14.4f}")

lz_overall = np.mean([results[s]["OVERALL"] for s in [42, 123, 456]])
os_overall = np.mean([os_results[s]["OVERALL"] for s in [42, 123, 456]])
print(f"{'OVERALL':<14} {lz_overall:14.4f} {os_overall:14.4f}")

print(f"\n  Teacher AUROC:               {t_auroc_lz:.4f}")
print(f"  TAMG-v2 pixels (learned Z):  {lz_overall:.4f}")
print(f"  Oracle-slot IWCM:            {os_overall:.4f}")
print(f"  Gap (pixels → oracle):       {os_overall - lz_overall:.4f}")

if lz_overall >= 0.85:
    print(f"\n  STRONG — 'from pixels' claim viable.")
elif lz_overall >= 0.75:
    print(f"\n  PROMISING — beats threshold, encoder improvement needed.")
elif lz_overall >= 0.65:
    print(f"\n  MARGINAL — signal present but weak. Better encoder required.")
else:
    print(f"\n  WEAK — learned Z insufficient for TAMG mining.")
