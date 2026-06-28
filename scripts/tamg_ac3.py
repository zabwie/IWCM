#!/usr/bin/env python3
"""TAMG-AC3: Full adversarial loop with learned corruptor network.

Corruptor MLP predicts per-slot perturbations to fool IWCM + validators.
Adversarial cycle: corruptor trains K steps, then IWCM trains K steps, repeat.
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

with open("data/compositional_grid.pkl", "rb") as f:
    data = pickle.load(f)
tv = [(torch.from_numpy(z0).float(), torch.from_numpy(A).float(), torch.from_numpy(Z).float())
      for z0, A, Z in data["train_valid"]]
test_valid = [(torch.from_numpy(z0).float(), torch.from_numpy(A).float(), torch.from_numpy(Z).float())
              for z0, A, Z in data["test_valid"]]
test_corr = [(torch.from_numpy(item[0][0]).float(), torch.from_numpy(item[0][1]).float(),
              torch.from_numpy(item[0][2]).float(), item[1]) for item in data["test_corr"]]
test_by_type = defaultdict(list)
for z0, A, Z, meta in test_corr:
    test_by_type[meta["violation_type"]].append((z0, A, Z))
N, d, H = MAX_OBJECTS, ORACLE_SLOT_DIM, 25

# ─── Learned Corruptor Network ────────────────────────────────────────────
class LearnedCorruptor(torch.nn.Module):
    """MLP that takes trajectory summary and outputs per-slot perturbations."""
    def __init__(self, d_slot, hidden=128):
        super().__init__()
        # Encode trajectory context: mean slot + z0
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(d_slot * 2, hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden, hidden),
        )
        # Predict per-slot perturbation
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(hidden, hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden, d_slot),
            torch.nn.Tanh(),  # bounded perturbations
        )

    def forward(self, Z, z0):
        B, H_in, N_in, d_in = Z.shape
        Z_mean = Z.mean(dim=1)  # (B, N, d)
        ctx = torch.cat([Z_mean, z0], dim=-1)  # (B, N, 2d)
        h = self.encoder(ctx)  # (B, N, hidden)
        delta = self.decoder(h) * 0.3  # (B, N, d), scale to 0.3
        # Apply perturbation to all timesteps for selected slots
        Z_c = Z.clone()
        for b in range(B):
            slot = np.random.randint(0, N_in)  # pick random slot
            t0 = np.random.randint(0, max(1, H_in // 2))
            Z_c[b, t0:, slot] += delta[b, slot].unsqueeze(0)
        return Z_c

# ─── Validators (reuse v3 architecture) ───────────────────────────────────
class V1(Validator): name = "v1_local"
class V2(Validator): name = "v2_velocity"
class V3(Validator): name = "v3_global"
class V4(Validator): name = "v4_cross"
class V5(Validator): name = "v5_short"
class V6(Validator): name = "v6_gating"

validators = [V1(), V2(), V3(), V4(), V5(), V6()]

validators[0].mlp = torch.nn.Sequential(torch.nn.Linear(d*2,128),torch.nn.ReLU(),torch.nn.Linear(128,1))

def v1f(self, Z, z0=None, A=None):
    return self.mlp(torch.cat(
        [Z[:,:-1].reshape(-1,N,d), Z[:,1:].reshape(-1,N,d)], dim=-1
    )).squeeze(-1).reshape(Z.shape[0],-1,N).mean(dim=(-2,-1))

validators[0].forward = types.MethodType(v1f, validators[0])

validators[1].mlp = torch.nn.Sequential(torch.nn.Linear(d,64),torch.nn.ReLU(),torch.nn.Linear(64,1))

def v2f(self, Z, z0=None, A=None):
    return self.mlp(
        (Z[:,1:]-Z[:,:-1]).reshape(-1,N,d)
    ).squeeze(-1).reshape(Z.shape[0],-1,N).mean(dim=(-2,-1))

validators[1].forward = types.MethodType(v2f, validators[1])

validators[2].gru = torch.nn.GRU(d,64,bidirectional=True,batch_first=True)
validators[2].scorer = torch.nn.Linear(128,1)

def v3f(self, Z, z0=None, A=None):
    B = Z.shape[0]
    _, h = self.gru(Z.reshape(B,-1,d))
    return self.scorer(torch.cat([h[0],h[1]],dim=-1)).squeeze(-1)

validators[2].forward = types.MethodType(v3f, validators[2])

validators[3].proj = torch.nn.Linear(d, 32)
validators[3].scorer = torch.nn.Linear(32, 1)

def v4f(self, Z, z0=None, A=None):
    B, H, N_in, d_in = Z.shape
    half = H // 2
    if half < 1:
        return torch.zeros(B, device=Z.device)
    return self.scorer(
        (self.proj(Z[:, :half].mean(dim=1)) - self.proj(Z[:, half:].mean(dim=1))).abs()
    ).squeeze(-1).mean(dim=-1)

validators[3].forward = types.MethodType(v4f, validators[3])

validators[4].mlp = torch.nn.Sequential(torch.nn.Linear(d*2,64),torch.nn.ReLU(),torch.nn.Linear(64,1))

def v5f(self, Z, z0=None, A=None):
    B, H, N_in, d_in = Z.shape
    h = min(5, H)
    if h < 2:
        return torch.zeros(B, device=Z.device)
    return self.mlp(torch.cat(
        [Z[:,:h-1].reshape(-1,N,d), Z[:,1:h].reshape(-1,N,d)], dim=-1
    )).squeeze(-1).reshape(B,h-1,N).mean(dim=(-2,-1))

validators[4].forward = types.MethodType(v5f, validators[4])

validators[5].gate = torch.nn.Sequential(torch.nn.Linear(d,1),torch.nn.Sigmoid())

def v6f(self, Z, z0=None, A=None):
    return self.gate(Z).squeeze(-1).mean(dim=(1,2))

validators[5].forward = types.MethodType(v6f, validators[5])

for v in validators: v.to(DEVICE)

# Train validators on valid data
params_v = [p for v in validators for p in v.parameters()]
opt_v = torch.optim.Adam(params_v, lr=1e-3)
print("Training validators...")
for ep in range(150):
    vi = np.random.choice(len(tv), 32, replace=False)
    Z_b = torch.stack([tv[i][2] for i in vi]).to(DEVICE)
    z0_b = torch.stack([tv[i][0] for i in vi]).to(DEVICE)
    A_b = torch.stack([tv[i][1] for i in vi]).to(DEVICE)
    loss = sum(F.binary_cross_entropy_with_logits(v(Z_b,z0_b,A_b), torch.ones(Z_b.shape[0],device=DEVICE)) for v in validators)
    opt_v.zero_grad(); loss.backward(); opt_v.step()
    if (ep+1)%75==0: print(f"  ep{ep+1}: loss={loss.item():.4f}")
for v in validators: v.eval()
print("Done.")

# ─── AC3 Adversarial Loop ────────────────────────────────────────────────
print("\nAC3 Adversarial Loop...")
corruptor = LearnedCorruptor(d, hidden=128).to(DEVICE)
iwcm = SlotIWCMEnergy(d_slot=d, d_action=11, hidden_dim=128, num_slots=N).to(DEVICE)
opt_corr = torch.optim.Adam(corruptor.parameters(), lr=1e-4)
opt_iwcm = torch.optim.Adam(iwcm.parameters(), lr=3e-3)

def validate_disagree(Z_b, z0_b, A_b):
    probs = [torch.sigmoid(v(Z_b,z0_b,A_b)) for v in validators]
    return torch.stack(probs, dim=0).var(dim=0).mean()

N_CYCLES = 5
STEPS_CORR = 50
STEPS_IWCM = 100
BATCH = 24

for cycle in range(N_CYCLES):
    print(f"\nCycle {cycle+1}/{N_CYCLES}")
    
    for param in validators:
        for p in param.parameters(): p.requires_grad = False
    
    for _ in range(STEPS_CORR):
        for v in validators: v.train()  # GRU needs training mode for backward
        vi = np.random.choice(len(tv), BATCH, replace=False)
        vZ = torch.stack([tv[i][2] for i in vi]).to(DEVICE)
        vz0 = torch.stack([tv[i][0] for i in vi]).to(DEVICE)
        vA = torch.stack([tv[i][1] for i in vi]).to(DEVICE)
        
        cZ = corruptor(vZ, vz0)
        
        accept = torch.sigmoid(-iwcm(vz0, vA, cZ))
        disagree = validate_disagree(cZ, vz0, vA)
        edit_cost = F.mse_loss(cZ, vZ)
        
        loss_c = -(accept.mean() + 0.5 * disagree - 0.1 * edit_cost)
        opt_corr.zero_grad()
        loss_c.backward()
        torch.nn.utils.clip_grad_norm_(corruptor.parameters(), 1.0)
        opt_corr.step()
    
    for _ in range(STEPS_IWCM):
        for v in validators: v.eval()
        vi = np.random.choice(len(tv), BATCH, replace=False)
        vZ = torch.stack([tv[i][2] for i in vi]).to(DEVICE)
        vz0 = torch.stack([tv[i][0] for i in vi]).to(DEVICE)
        vA = torch.stack([tv[i][1] for i in vi]).to(DEVICE)
        
        with torch.no_grad():
            cZ = corruptor(vZ, vz0)
            w = validate_disagree(cZ, vz0, vA).clamp(0.1, 2.0)
        
        ev = iwcm(vz0, vA, vZ)
        ec = iwcm(vz0, vA, cZ)
        loss_i = F.relu(ev+1).mean() + (w*F.relu(1-ec)).mean() + 0.001*(ev.pow(2).mean()+ec.pow(2).mean())
        opt_iwcm.zero_grad(); loss_i.backward()
        torch.nn.utils.clip_grad_norm_(iwcm.parameters(), 1.0); opt_iwcm.step()
    
    print(f"  corr: loss={loss_c.item():.3f} accept={accept.mean():.3f} disagree={disagree.item():.4f}")
    print(f"  iwcm: loss={loss_i.item():.3f} ev={ev.mean():+.2f} ec={ec.mean():+.2f}")

# ─── Evaluate ────────────────────────────────────────────────────────────
iwcm.eval()
print(f"\nCross-surface evaluation...")
with torch.no_grad():
    ve = [iwcm(z0.unsqueeze(0).to(DEVICE), A.unsqueeze(0).to(DEVICE),
               Z.unsqueeze(0).to(DEVICE)).item() for z0, A, Z in test_valid]
    per = {}
    for vt in sorted(test_by_type.keys()):
        ce = [iwcm(z0.unsqueeze(0).to(DEVICE), A.unsqueeze(0).to(DEVICE),
                   Z.unsqueeze(0).to(DEVICE)).item() for z0, A, Z in test_by_type[vt]]
        a = roc_auc_score([0]*len(ve)+[1]*len(ce), ve+ce); per[vt]=a
        print(f"  {vt:<14} {a:.3f}")
    avg = np.mean(list(per.values()))
    print(f"  {'AVERAGE':<14} {avg:.3f}")
    print(f"\nOracle: 0.962 | Random: 0.734 | TAMG v3: 0.765 | TAMG-AC3: {avg:.3f}")
