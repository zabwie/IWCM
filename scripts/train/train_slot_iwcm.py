#!/usr/bin/env python3
"""Train slot-aware IWCM on oracle slot data."""
import sys, torch, pickle, numpy as np, torch.nn.functional as F
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.seed import set_seed
from src.iwcm.slot_energy import SlotIWCMEnergy
from src.env.oracle_slot_encoder import ORACLE_SLOT_DIM, MAX_OBJECTS

set_seed(42)
device = "cuda" if torch.cuda.is_available() else "cpu"
N, d, H = MAX_OBJECTS, ORACLE_SLOT_DIM, 25
print(f"Device: {device}, slots={N}, dim={d}")

with open("data/oracle_slot_trajs_v2.pkl", "rb") as f:
    data = pickle.load(f)

def to_4d(trajs):
    result = []
    for z0, A, Z in trajs:
        result.append((
            torch.from_numpy(z0).float(),          # (N, d) → keep as (N, d) for z0
            torch.from_numpy(A).float(),             # (H, 11)
            torch.from_numpy(Z).float(),            # (H, N, d) — KEEP 4D!
        ))
    return result

valid = to_4d(data["valid"])
corr = to_4d(data["corruptions"])
print(f"Valid: {len(valid)}, Corruptions: {len(corr)}")

# Slot-aware energy function
efn = SlotIWCMEnergy(d_slot=d, d_action=11, hidden_dim=192, num_slots=N).to(device)
opt = torch.optim.Adam(efn.parameters(), lr=3e-3)
print(f"Params: {sum(p.numel() for p in efn.parameters()):,}")

NUM_EPOCHS = 200
BATCH = 32
MARGIN = 0.5
REG = 0.001

for epoch in range(NUM_EPOCHS):
    vi = np.random.choice(len(valid), min(BATCH, len(valid)), replace=False)
    ci = np.random.choice(len(corr), min(BATCH * 2, len(corr)), replace=False)

    vz0 = torch.stack([valid[i][0] for i in vi]).to(device)
    vA = torch.stack([valid[i][1] for i in vi]).to(device)
    vZ = torch.stack([valid[i][2] for i in vi]).to(device)
    cz0 = torch.stack([corr[i][0] for i in ci]).to(device)
    cA = torch.stack([corr[i][1] for i in ci]).to(device)
    cZ = torch.stack([corr[i][2] for i in ci]).to(device)

    opt.zero_grad()
    ev = efn(vz0, vA, vZ)
    ec = efn(cz0, cA, cZ)
    loss = F.relu(ev + MARGIN).mean() + F.relu(MARGIN - ec).mean() + REG * (ev.pow(2).mean() + ec.pow(2).mean())
    loss.backward()
    torch.nn.utils.clip_grad_norm_(efn.parameters(), 1.0)
    opt.step()

    if (epoch + 1) % 20 == 0:
        ph_v = efn.per_head(vz0[:1], vA[:1], vZ[:1])
        ph_c = efn.per_head(cz0[:1], cA[:1], cZ[:1])
        ph_str = " ".join(f"{n}={ph_v[n].item():.2f}/{ph_c[n].item():.2f}" for n in ph_v)
        print(f"Epoch {epoch+1}: loss={loss.item():.4f} E_valid={ev.mean().item():.3f} "
              f"E_invalid={ec.mean().item():.3f} gap={ec.mean().item()-ev.mean().item():.3f}")
        print(f"  Heads: {ph_str}")

# Save
torch.save(efn.state_dict(), "outputs/checkpoints/slot_iwcm_energy_v2_perobject.pt")
print("Saved to outputs/checkpoints/slot_iwcm_energy_v2_perobject.pt")

# Quick eval
efn.eval()
ev_list = [efn(valid[i][0].to(device).unsqueeze(0), valid[i][1].to(device).unsqueeze(0),
               valid[i][2].to(device).unsqueeze(0)).item() for i in range(min(50, len(valid)))]
ec_list = [efn(corr[i][0].to(device).unsqueeze(0), corr[i][1].to(device).unsqueeze(0),
               corr[i][2].to(device).unsqueeze(0)).item() for i in range(min(100, len(corr)))]
print(f"\nFinal: E_valid={np.mean(ev_list):.3f}+/-{np.std(ev_list):.3f} "
      f"E_invalid={np.mean(ec_list):.3f}+/-{np.std(ec_list):.3f} "
      f"gap={np.mean(ec_list)-np.mean(ev_list):.3f}")
print(f"Accept valid: {np.mean([torch.sigmoid(torch.tensor(-e)).item() for e in ev_list]):.3f}")
print(f"Accept invalid: {np.mean([torch.sigmoid(torch.tensor(-e)).item() for e in ec_list]):.3f}")
