#!/usr/bin/env python3
"""Speed + scaling + accuracy Pareto benchmark for all models."""
import sys, torch, pickle, numpy as np, torch.nn.functional as F, time
from collections import defaultdict
sys.path.insert(0, '.')
from src.utils.seed import set_seed
from src.iwcm.slot_energy import SlotIWCMEnergy
from src.env.oracle_slot_encoder import ORACLE_SLOT_DIM, MAX_OBJECTS

H, N, d = 25, MAX_OBJECTS, ORACLE_SLOT_DIM
device = "cuda" if torch.cuda.is_available() else "cpu"

with open("data/compositional_grid.pkl", "rb") as f:
    grid = pickle.load(f)


class FlatMLP(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(H * N * d, 256), torch.nn.ReLU(),
            torch.nn.Linear(256, 64), torch.nn.ReLU(), torch.nn.Linear(64, 1))

    def forward(self, z0, A, Z):
        return self.net(Z.reshape(Z.shape[0], -1)).squeeze(-1)


class SlotTransformer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = torch.nn.Linear(d, 64)
        enc = torch.nn.TransformerEncoderLayer(d_model=64, nhead=4, dim_feedforward=128,
                                                dropout=0.1, batch_first=True)
        self.transformer = torch.nn.TransformerEncoder(enc, num_layers=2)
        self.scorer = torch.nn.Linear(64, 1)

    def forward(self, z0, A, Z):
        B = Z.shape[0]
        Zp = self.proj(Z).reshape(B, H * N, 64)
        return self.scorer(self.transformer(Zp).mean(dim=1)).squeeze(-1)


@torch.no_grad()
def benchmark(model, batch_size, steps=50, warmup=10):
    model.eval()
    model.to(device)
    z0 = torch.randn(batch_size, N, d, device=device)
    A = torch.randn(batch_size, H, 11, device=device)
    Z = torch.randn(batch_size, H, N, d, device=device)
    for _ in range(warmup):
        _ = model(z0, A, Z)
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(steps):
        _ = model(z0, A, Z)
    end.record()
    torch.cuda.synchronize()
    ms_total = start.elapsed_time(end)
    ms_per_batch = ms_total / steps
    ms_per_sample = ms_per_batch / batch_size
    samples_per_sec = 1000 / ms_per_sample
    torch.cuda.reset_peak_memory_stats()
    _ = model(z0, A, Z)
    vram_mb = torch.cuda.max_memory_allocated() / 1024**2
    params = sum(p.numel() for p in model.parameters())
    return {"ms_per_batch": ms_per_batch, "ms_per_sample": ms_per_sample,
            "samples_per_sec": samples_per_sec, "vram_mb": vram_mb, "params": params}


def sd(e):
    return [(torch.from_numpy(z0).float(), torch.from_numpy(A).float(),
             torch.from_numpy(Z).float()) for z0, A, Z in e]


tv = sd(grid["train_valid"])
tcm = [(torch.from_numpy(e[0]).float(), torch.from_numpy(e[1]).float(),
        torch.from_numpy(e[2]).float(), m) for e, m in grid["train_corr"]]
tev = sd(grid["test_valid"])
tecm = [(torch.from_numpy(e[0]).float(), torch.from_numpy(e[1]).float(),
         torch.from_numpy(e[2]).float(), m) for e, m in grid["test_corr"]]


def train_and_eval(ModelClass, name, epochs=120, lr=3e-4, seed=42):
    set_seed(seed)
    model = ModelClass().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    t0 = time.time()
    for ep in range(epochs):
        vi = np.random.choice(len(tv), min(16, len(tv)), replace=False)
        ci = np.random.choice(len(tcm), min(32, len(tcm)), replace=False)
        vz0 = torch.stack([tv[i][0] for i in vi]).to(device)
        vA = torch.stack([tv[i][1] for i in vi]).to(device)
        vZ = torch.stack([tv[i][2] for i in vi]).to(device)
        cz0 = torch.stack([tcm[i][0] for i in ci]).to(device)
        cA = torch.stack([tcm[i][1] for i in ci]).to(device)
        cZ = torch.stack([tcm[i][2] for i in ci]).to(device)
        opt.zero_grad()
        ev = model(vz0, vA, vZ); ec = model(cz0, cA, cZ)
        loss = (F.relu(ev + 1.0).mean() + F.relu(1.0 - ec).mean() +
                0.001 * (ev.pow(2).mean() + ec.pow(2).mean()))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
    train_time = time.time() - t0

    pl = defaultdict(list); va = []; ir = []
    for vz, vA, vZ in tev[:80]:
        with torch.no_grad():
            s = torch.sigmoid(-model(vz.unsqueeze(0).to(device),
                                      vA.unsqueeze(0).to(device),
                                      vZ.unsqueeze(0).to(device))).item()
        va.append(s > 0.5)
    for cz, cA, cZ, meta in tecm:
        with torch.no_grad():
            s = torch.sigmoid(-model(cz.unsqueeze(0).to(device),
                                      cA.unsqueeze(0).to(device),
                                      cZ.unsqueeze(0).to(device))).item()
        ir.append(s < 0.5)
        pl[meta["law_type"]].append(s < 0.5)
    acc = {"valid_acc": np.mean(va), "invalid_rej": np.mean(ir),
           "conservation": np.mean(pl.get("conservation", [0.])),
           "identity": np.mean(pl.get("identity", [0.]))}
    return acc, train_time


# Benchmark
print("=" * 85)
print(f"{'Model':<25} {'Params':>8} {'Samples/s':>10} {'ms/sample':>10} {'VRAM MB':>8} {'Cons':>8} {'Id':>8} {'Valid':>8} {'Rej':>8}")
print("-" * 85)

models = {"MLP": FlatMLP, "SlotTransformer": SlotTransformer, "IWCM": lambda: SlotIWCMEnergy(d_slot=d, d_action=11, hidden_dim=128, num_slots=N)}

for name, cls in models.items():
    model = cls()
    bench = benchmark(model, 64)
    acc, ttime = train_and_eval(cls, name, epochs=80)
    print(f"{name:<25} {bench['params']:>8,} {bench['samples_per_sec']:>10.0f} "
          f"{bench['ms_per_sample']:>10.3f} {bench['vram_mb']:>8.1f} "
          f"{acc['conservation']:>8.3f} {acc['identity']:>8.3f} "
          f"{acc['valid_acc']:>8.3f} {acc['invalid_rej']:>8.3f}")

# Scaling benchmark with H
print("\n" + "-" * 85)
print("SCALING WITH HORIZON (MLP vs IWCM)")
print("-" * 85)
for H_test in [5, 10, 20, 50]:
    d_tmp = H_test * N * d
    z0_s = torch.randn(64, N, d, device=device)
    A_s = torch.randn(64, H_test, 11, device=device)
    Z_s = torch.randn(64, H_test, N, d, device=device)
    b_mlp = benchmark(FlatMLP(), 64, steps=30, warmup=5)
    b_iwcm = benchmark(SlotIWCMEnergy(d_slot=d, d_action=11, hidden_dim=128, num_slots=N), 64, steps=30, warmup=5)
    print(f"H={H_test:<4} MLP={b_mlp['ms_per_sample']:.3f}ms/sample  IWCM={b_iwcm['ms_per_sample']:.3f}ms/sample  "
          f"ratio={b_iwcm['ms_per_sample']/max(b_mlp['ms_per_sample'],0.001):.1f}x")

print("\n" + "=" * 85)
print("ACCURACY PER MILLISECOND (higher = better)")
print("=" * 85)
for name, cls in models.items():
    model = cls()
    bench = benchmark(model, 64)
    acc, _ = train_and_eval(cls, name, epochs=80, seed=456)
    acc_per_ms = acc["conservation"] / max(bench["ms_per_sample"], 0.001)
    print(f"{name}: {acc_per_ms:.1f} cons/ms")
