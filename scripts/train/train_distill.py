#!/usr/bin/env python3
"""Teacher-student distillation: IWCM teacher → fast MLP student."""
import sys, torch, pickle, numpy as np, torch.nn.functional as F
from collections import defaultdict
sys.path.insert(0, '.')
from src.utils.seed import set_seed
from src.iwcm.slot_energy import SlotIWCMEnergy
from src.env.oracle_slot_encoder import ORACLE_SLOT_DIM, MAX_OBJECTS

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


class StudentMLP(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(H * N * d, 256), torch.nn.ReLU(),
            torch.nn.Linear(256, 128), torch.nn.ReLU(),
            torch.nn.Linear(128, 1))

    def forward(self, z0, A, Z):
        return self.net(Z.reshape(Z.shape[0], -1)).squeeze(-1)


for seed in [42, 123, 456]:
    set_seed(seed)

    # Train IWCM teacher
    teacher = SlotIWCMEnergy(d_slot=d, d_action=11, hidden_dim=128, num_slots=N).to(device)
    opt = torch.optim.Adam(teacher.parameters(), lr=3e-4)
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
        ev = teacher(vz0, vA, vZ)
        ec = teacher(cz0, cA, cZ)
        loss = (F.relu(ev + 1.0).mean() + F.relu(1.0 - ec).mean() +
                0.001 * (ev.pow(2).mean() + ec.pow(2).mean()))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(teacher.parameters(), 1.0)
        opt.step()
    teacher.eval()

    # Distill into student MLP
    student = StudentMLP().to(device)
    opt_s = torch.optim.Adam(student.parameters(), lr=1e-3)
    for ep in range(200):
        vi = np.random.choice(len(tv), min(32, len(tv)), replace=False)
        ci = np.random.choice(len(tcm), min(64, len(tcm)), replace=False)
        vz0 = torch.stack([tv[i][0] for i in vi]).to(device)
        cz0 = torch.stack([tcm[i][0] for i in ci]).to(device)
        vA = torch.stack([tv[i][1] for i in vi]).to(device)
        cA = torch.stack([tcm[i][1] for i in ci]).to(device)
        vZ = torch.stack([tv[i][2] for i in vi]).to(device)
        cZ = torch.stack([tcm[i][2] for i in ci]).to(device)
        with torch.no_grad():
            ev_t = teacher(vz0, vA, vZ)
            ec_t = teacher(cz0, cA, cZ)
        opt_s.zero_grad()
        ev_s = student(vz0, vA, vZ)
        ec_s = student(cz0, cA, cZ)
        loss = (F.relu(ev_s + 1.0).mean() + F.relu(1.0 - ec_s).mean() +
                0.5 * (F.mse_loss(ev_s, ev_t) + F.mse_loss(ec_s, ec_t)))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
        opt_s.step()

    # Evaluate student
    pl = defaultdict(list)
    va = []
    ir = []
    for vz, vA, vZ in tev[:80]:
        s = torch.sigmoid(-student(vz.unsqueeze(0).to(device),
                                    vA.unsqueeze(0).to(device),
                                    vZ.unsqueeze(0).to(device))).item()
        va.append(s > 0.5)
    for cz, cA, cZ, meta in tecm:
        s = torch.sigmoid(-student(cz.unsqueeze(0).to(device),
                                    cA.unsqueeze(0).to(device),
                                    cZ.unsqueeze(0).to(device))).item()
        ir.append(s < 0.5)
        pl[meta["law_type"]].append(s < 0.5)
    print(f"Seed {seed}: valid={np.mean(va):.3f} rej={np.mean(ir):.3f} "
          f"cons={np.mean(pl.get('conservation', [0.])):.3f} "
          f"ident={np.mean(pl.get('identity', [0.])):.3f}")
