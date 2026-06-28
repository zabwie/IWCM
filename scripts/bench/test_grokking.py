#!/usr/bin/env python3
"""Grokking test: same-param MLP vs Pooling IWCM with train/val curves."""
import sys, torch, pickle, numpy as np, torch.nn.functional as F
from collections import defaultdict
sys.path.insert(0, '.')
from src.utils.seed import set_seed
from src.iwcm.fused_energy import FusedIWCMEnergy
from src.encoder.oracle_slot_encoder import ORACLE_SLOT_DIM, MAX_OBJECTS

H, N, d = 25, MAX_OBJECTS, ORACLE_SLOT_DIM
device = "cuda"

with open("data/compositional_grid.pkl", "rb") as f:
    grid = pickle.load(f)


def sd(e):
    return [(torch.from_numpy(z0).float(), torch.from_numpy(A).float(),
             torch.from_numpy(Z).float()) for z0, A, Z in e]


tv = sd(grid["train_valid"])
tev = sd(grid["test_valid"])
tcm = [(torch.from_numpy(e[0]).float(), torch.from_numpy(e[1]).float(),
        torch.from_numpy(e[2]).float(), m) for e, m in grid["train_corr"]]
tecm = [(torch.from_numpy(e[0]).float(), torch.from_numpy(e[1]).float(),
         torch.from_numpy(e[2]).float(), m) for e, m in grid["test_corr"]]


class SmallMLP(torch.nn.Module):
    """MLP with same param count as Pooling IWCM (~54K)."""
    def __init__(self):
        super().__init__()
        # 54K params: input(3800) -> 14 hidden -> 14 -> 1
        self.net = torch.nn.Sequential(
            torch.nn.Linear(H * N * d, 14), torch.nn.ReLU(),
            torch.nn.Linear(14, 14), torch.nn.ReLU(),
            torch.nn.Linear(14, 1))

    def forward(self, z0, A, Z):
        return self.net(Z.reshape(Z.shape[0], -1)).squeeze(-1)


def eval_model(model, is_iwcm=False):
    pl = defaultdict(list)
    va, ir = [], []
    for vz, vA, vZ in tev[:80]:
        if is_iwcm:
            s = model.score_acceptance(vz.unsqueeze(0).to(device),
                                        vA.unsqueeze(0).to(device),
                                        vZ.unsqueeze(0).to(device)).item()
        else:
            s = torch.sigmoid(-model(vz.unsqueeze(0).to(device),
                                      vA.unsqueeze(0).to(device),
                                      vZ.unsqueeze(0).to(device))).item()
        va.append(s > 0.5)
    for cz, cA, cZ, meta in tecm:
        if is_iwcm:
            s = model.score_acceptance(cz.unsqueeze(0).to(device),
                                        cA.unsqueeze(0).to(device),
                                        cZ.unsqueeze(0).to(device)).item()
        else:
            s = torch.sigmoid(-model(cz.unsqueeze(0).to(device),
                                      cA.unsqueeze(0).to(device),
                                      cZ.unsqueeze(0).to(device))).item()
        ir.append(s < 0.5)
        pl[meta["law_type"]].append(s < 0.5)
    return {"valid": np.mean(va), "rej": np.mean(ir),
            "cons": np.mean(pl.get("conservation", [0.])),
            "ident": np.mean(pl.get("identity", [0.]))}


# Train Small MLP
print("=== Small MLP (54K params) ===\n")
for seed in [42, 123, 456]:
    set_seed(seed)
    model = SmallMLP().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=3e-4)
    for ep in range(200):
        vi = np.random.choice(len(tv), min(16, len(tv)), replace=False)
        ci = np.random.choice(len(tcm), min(32, len(tcm)), replace=False)
        vz0 = torch.stack([tv[i][0] for i in vi]).to(device)
        cz0 = torch.stack([tcm[i][0] for i in ci]).to(device)
        vA = torch.stack([tv[i][1] for i in vi]).to(device)
        cA = torch.stack([tcm[i][1] for i in ci]).to(device)
        vZ = torch.stack([tv[i][2] for i in vi]).to(device)
        cZ = torch.stack([tcm[i][2] for i in ci]).to(device)
        opt.zero_grad()
        ev = model(vz0, vA, vZ)
        ec = model(cz0, cA, cZ)
        loss = (F.relu(ev + 1.0).mean() + F.relu(1.0 - ec).mean() +
                0.001 * (ev.pow(2).mean() + ec.pow(2).mean()))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
    r = eval_model(model, is_iwcm=False)
    print(f"Seed {seed}: {r}")

# Also check Pooling IWCM for grokking — track train/val over epochs
print("\n=== Pooling IWCM — Train/Val Curves ===")
set_seed(42)
iwcm = FusedIWCMEnergy(d_slot=d, d_action=11, hidden=128, num_slots=N).to(device)
opt_i = torch.optim.Adam(iwcm.parameters(), lr=3e-4)

for ep in range(300):
    vi = np.random.choice(len(tv), min(16, len(tv)), replace=False)
    ci = np.random.choice(len(tcm), min(32, len(tcm)), replace=False)
    vz0 = torch.stack([tv[i][0] for i in vi]).to(device)
    cz0 = torch.stack([tcm[i][0] for i in ci]).to(device)
    vA = torch.stack([tv[i][1] for i in vi]).to(device)
    cA = torch.stack([tcm[i][1] for i in ci]).to(device)
    vZ = torch.stack([tv[i][2] for i in vi]).to(device)
    cZ = torch.stack([tcm[i][2] for i in ci]).to(device)
    opt_i.zero_grad()
    ev = iwcm(vz0, vA, vZ)
    ec = iwcm(cz0, cA, cZ)
    train_loss = (F.relu(ev + 1.0).mean() + F.relu(1.0 - ec).mean())
    reg = 0.001 * (ev.pow(2).mean() + ec.pow(2).mean())
    (train_loss + reg).backward()
    torch.nn.utils.clip_grad_norm_(iwcm.parameters(), 1.0)
    opt_i.step()

    if (ep + 1) % 50 == 0:
        r = eval_model(iwcm, is_iwcm=True)
        print(f"Epoch {ep+1}: train_loss={train_loss.item():.4f} "
              f"val_cons={r['cons']:.3f} val_ident={r['ident']:.3f} rej={r['rej']:.3f}")

print("\nFinal comparison:")
for name, model_fn, iwcm_flag in [("54K MLP", SmallMLP, False), ("54K IWCM", lambda: FusedIWCMEnergy(d_slot=d, d_action=11, hidden=128, num_slots=N), True)]:
    for seed in [42, 123, 456]:
        set_seed(seed)
        m = model_fn().to(device)
        opt = torch.optim.Adam(m.parameters(), lr=3e-4)
        for ep in range(200):
            vi = np.random.choice(len(tv), min(16, len(tv)), replace=False)
            ci = np.random.choice(len(tcm), min(32, len(tcm)), replace=False)
            vz0 = torch.stack([tv[i][0] for i in vi]).to(device)
            cz0 = torch.stack([tcm[i][0] for i in ci]).to(device)
            vA = torch.stack([tv[i][1] for i in vi]).to(device)
            cA = torch.stack([tcm[i][1] for i in ci]).to(device)
            vZ = torch.stack([tv[i][2] for i in vi]).to(device)
            cZ = torch.stack([tcm[i][2] for i in ci]).to(device)
            opt.zero_grad()
            if iwcm_flag:
                ev = m(vz0, vA, vZ); ec = m(cz0, cA, cZ)
            else:
                ev = m(vz0, vA, vZ); ec = m(cz0, cA, cZ)
            loss = (F.relu(ev + 1.0).mean() + F.relu(1.0 - ec).mean() +
                    0.001 * (ev.pow(2).mean() + ec.pow(2).mean()))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
            opt.step()
        r = eval_model(m, is_iwcm=iwcm_flag)
        avg_cons = np.mean([r["cons"]])
        print(f"{name} seed {seed}: {r}")
        break
