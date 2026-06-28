#!/usr/bin/env python3
"""TAMG v4: Genuine information asymmetry — each validator sees a DIFFERENT input projection.

V1: raw slots (local consistency)
V2: slot differences / velocity (dynamics)
V3: first-half vs second-half (global structure)
V4: slot norms / energy (physics-like)
V5: slot cross-correlations (identity consistency)
V6: existence flags only (object permanence)

Key insight: different INPUTS create genuine disagreement. A perturbation
that looks fine to the velocity validator may look wrong to the energy validator.
"""
import sys, pickle, numpy as np, torch, torch.nn.functional as F, types
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.seed import set_seed
from src.iwcm.slot_energy import SlotIWCMEnergy
from src.tamg.validators.committee import Validator
from src.encoder.oracle_slot_encoder import ORACLE_SLOT_DIM, MAX_OBJECTS
from collections import defaultdict
from sklearn.metrics import roc_auc_score

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
set_seed(42)

with open("data/compositional_grid.pkl", "rb") as f: data = pickle.load(f)
tv = [(torch.from_numpy(z0).float(), torch.from_numpy(A).float(), torch.from_numpy(Z).float())
      for z0, A, Z in data["train_valid"]]
test_valid = [(torch.from_numpy(z0).float(), torch.from_numpy(A).float(), torch.from_numpy(Z).float())
              for z0, A, Z in data["test_valid"]]
test_corr = [(torch.from_numpy(item[0][0]).float(), torch.from_numpy(item[0][1]).float(),
              torch.from_numpy(item[0][2]).float(), item[1]) for item in data["test_corr"]]
test_by_type = defaultdict(list)
for z0, A, Z, meta in test_corr: test_by_type[meta["violation_type"]].append((z0, A, Z))
N, d, H = MAX_OBJECTS, ORACLE_SLOT_DIM, 25

# ─── Validators with GENUINELY DIFFERENT INPUTS ──────────────────────────

class VInput1(Validator):
    """Raw slots — local transition consistency."""
    name = "v1_raw"
    def __init__(self, d):
        super().__init__()
        self.mlp = torch.nn.Sequential(torch.nn.Linear(d*2, 128), torch.nn.ReLU(), torch.nn.Linear(128, 1))
    def project(self, Z):
        return torch.cat([Z[:,:-1].reshape(-1,N,d), Z[:,1:].reshape(-1,N,d)], dim=-1)
    def forward(self, Z, z0=None, A=None):
        x = self.project(Z); B = Z.shape[0]
        return self.mlp(x).squeeze(-1).reshape(B, -1, N).mean(dim=(-2, -1))

class VInput2(Validator):
    """Velocity — frame differences."""
    name = "v2_velocity"
    def __init__(self, d):
        super().__init__()
        self.mlp = torch.nn.Sequential(torch.nn.Linear(d, 64), torch.nn.ReLU(), torch.nn.Linear(64, 1))
    def project(self, Z):
        return (Z[:,1:] - Z[:,:-1]).reshape(-1, N, d)  # velocity
    def forward(self, Z, z0=None, A=None):
        x = self.project(Z); B = Z.shape[0]
        return self.mlp(x).squeeze(-1).reshape(B, -1, N).mean(dim=(-2, -1))

class VInput3(Validator):
    """Global — first half vs second half contrast."""
    name = "v3_global"
    def __init__(self, d):
        super().__init__()
        self.proj = torch.nn.Linear(d, 32); self.scorer = torch.nn.Linear(64, 1)
    def project(self, Z):
        B, H, N_in, d_in = Z.shape; half = H//2
        if half < 1: return torch.zeros(B, N_in, 64, device=Z.device)
        early = self.proj(Z[:,:half].mean(dim=1))
        late = self.proj(Z[:,half:].mean(dim=1))
        return torch.cat([early, late], dim=-1)  # (B, N, 64)
    def forward(self, Z, z0=None, A=None):
        return self.scorer(self.project(Z)).squeeze(-1).mean(dim=-1)

class VInput4(Validator):
    """Energy — slot L2 norms (physics-like invariant)."""
    name = "v4_energy"
    def __init__(self, d):
        super().__init__()
        self.mlp = torch.nn.Sequential(torch.nn.Linear(1, 32), torch.nn.ReLU(), torch.nn.Linear(32, 1))
    def project(self, Z):
        return Z.pow(2).sum(dim=-1, keepdim=True)  # (B, H, N, 1) — per-slot energy
    def forward(self, Z, z0=None, A=None):
        x = self.project(Z); B = Z.shape[0]
        # Check energy stability across time
        diff = (x[:,1:] - x[:,:-1]).abs().reshape(-1, N, 1)
        return self.mlp(diff).squeeze(-1).reshape(B, -1, N).mean(dim=(-2, -1))

class VInput5(Validator):
    """Identity — slot cross-correlation across time."""
    name = "v5_identity"
    def __init__(self, d):
        super().__init__()
        self.mlp = torch.nn.Sequential(torch.nn.Linear(1, 32), torch.nn.ReLU(), torch.nn.Linear(32, 1))
    def project(self, Z):
        B, H, N_in, d_in = Z.shape
        if H < 2: return torch.zeros(B, N_in, 1, device=Z.device)
        # Cosine similarity between consecutive frames per slot
        z_t = F.normalize(Z[:,:-1], dim=-1)
        z_tp1 = F.normalize(Z[:,1:], dim=-1)
        sim = (z_t * z_tp1).sum(dim=-1, keepdim=True)  # (B, H-1, N, 1)
        return sim
    def forward(self, Z, z0=None, A=None):
        x = self.project(Z); B = Z.shape[0]
        return self.mlp(x.reshape(-1, N, 1)).squeeze(-1).reshape(B, -1, N).mean(dim=(-2, -1))

class VInput6(Validator):
    """Existence — only channel 0 (existence flag)."""
    name = "v6_existence"
    def __init__(self, d):
        super().__init__()
        self.mlp = torch.nn.Sequential(torch.nn.Linear(1, 16), torch.nn.ReLU(), torch.nn.Linear(16, 1))
    def project(self, Z):
        return Z[..., 0:1]  # (B, H, N, 1) — only existence channel
    def forward(self, Z, z0=None, A=None):
        x = self.project(Z); B = Z.shape[0]
        return self.mlp(x.reshape(-1, 1)).squeeze(-1).reshape(B, -1, N).mean(dim=(-2, -1))

validators = [
    VInput1(d).to(DEVICE), VInput2(d).to(DEVICE), VInput3(d).to(DEVICE),
    VInput4(d).to(DEVICE), VInput5(d).to(DEVICE), VInput6(d).to(DEVICE),
]
print(f"Built {len(validators)} validators with DIFFERENT inputs")

# Train validators on valid data
params_v = [p for v in validators for p in v.parameters()]
opt_v = torch.optim.Adam(params_v, lr=1e-3)
for ep in range(150):
    vi = np.random.choice(len(tv), 32, replace=False)
    Z_b = torch.stack([tv[i][2] for i in vi]).to(DEVICE)
    loss = sum(F.binary_cross_entropy_with_logits(v(Z_b), torch.ones(Z_b.shape[0],device=DEVICE)) for v in validators)
    opt_v.zero_grad(); loss.backward(); opt_v.step()
    if (ep+1)%75==0: print(f"  ep{ep+1}: loss={loss.item():.4f}")
for v in validators: v.eval()

# ─── Measure baseline disagreement on valid vs corrupt ───────────────────
print("\nBaseline disagreement (valid vs corrupt):")
def disagreement(Z_b):
    probs = [torch.sigmoid(v(Z_b)) for v in validators]
    return torch.stack(probs, dim=0).var(dim=0).mean()

Z_v = torch.stack([tv[i][2] for i in range(50)]).to(DEVICE)
d_v = disagreement(Z_v)

for vt in sorted(test_by_type.keys()):
    items = test_by_type[vt][:50]
    Z_c = torch.stack([items[i][2] for i in range(len(items))]).to(DEVICE)
    d_c = disagreement(Z_c)
    print(f"  {vt:<14} d_valid={d_v.item():.4f} d_corrupt={d_c.item():.4f} ratio={d_c.item()/max(d_v.item(),1e-5):.2f}x")

# ─── Simple corruption generation + self-supervised IWCM ────────────────
print("\nSelf-supervised IWCM (disagreement-weighted)...")
# Use KMeans centroids for corruption generation
from sklearn.cluster import KMeans
diffs_list = []
for z0, A, Z in tv[:50]:
    for t in range(H-1):
        for n in range(N):
            if Z[t,n].sum()>0.1: diffs_list.append((Z[t+1,n]-Z[t,n]).numpy())
diffs_arr = np.stack(diffs_list)
n_ops = 8
kmeans = KMeans(n_clusters=n_ops, random_state=42, n_init=10).fit(diffs_arr)
centroids = torch.from_numpy(kmeans.cluster_centers_).float().to(DEVICE)

model = SlotIWCMEnergy(d_slot=d, d_action=11, hidden_dim=128, num_slots=N).to(DEVICE)
opt = torch.optim.Adam(model.parameters(), lr=3e-3)

for ep in range(400):
    batch = min(32, len(tv))
    vi = np.random.choice(len(tv), batch, replace=False)
    vZ = torch.stack([tv[i][2] for i in vi]).to(DEVICE)
    vz0 = torch.stack([tv[i][0] for i in vi]).to(DEVICE)
    vA = torch.stack([tv[i][1] for i in vi]).to(DEVICE)
    
    cZ = vZ.clone()
    for b in range(batch):
        cZ[b, np.random.randint(0,max(1,H//2)):, np.random.randint(0,N)] += centroids[np.random.randint(0,n_ops)] * 0.4
    
    d_c = disagreement(cZ)
    d_v = disagreement(vZ)
    w = (d_c / (d_v + 1e-5)).detach().clamp(0.5, 3.0)
    
    ev = model(vz0, vA, vZ); ec = model(vz0, vA, cZ)
    loss = F.relu(ev+1).mean() + (w*F.relu(1-ec)).mean() + 0.001*(ev.pow(2).mean()+ec.pow(2).mean())
    opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
    if (ep+1)%100==0:
        print(f"  ep{ep+1}: ev={ev.mean():+.2f} ec={ec.mean():+.2f} d_ratio={w.mean():.3f}")

# ─── Evaluate ────────────────────────────────────────────────────────────
model.eval()
ve = [model(z0.unsqueeze(0).to(DEVICE),A.unsqueeze(0).to(DEVICE),Z.unsqueeze(0).to(DEVICE)).item() for z0,A,Z in test_valid]
per = {}
for vt in sorted(test_by_type.keys()):
    ce = [model(z0.unsqueeze(0).to(DEVICE),A.unsqueeze(0).to(DEVICE),Z.unsqueeze(0).to(DEVICE)).item() for z0,A,Z in test_by_type[vt]]
    a=roc_auc_score([0]*len(ve)+[1]*len(ce), ve+ce); per[vt]=a
    print(f"  {vt:<14} {a:.3f}")
avg = np.mean(list(per.values()))
print(f"  AVERAGE         {avg:.3f}")
print(f"\nOracle:0.962 Random:0.734 TAMGv3:0.765 TAMGv4:{avg:.3f}")
