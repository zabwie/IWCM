#!/usr/bin/env python3
"""Energy vs physical MSE correlation on DM Control cartpole.

Produces LaTeX table:
  H  | Rollout MSE | z0-rep MSE | Warm MSE | Rollout ΔE | z0-rep ΔE | Warm ΔE

ponytail: reuse _common.py helpers, do not duplicate.
"""
from _common import *
import torch.nn.functional as F

w, e, da = env()
tv, tc, ts = data(w, e)
print(f"Train {len(tv)} valid + {len(tc)} corrupted, test {len(ts)} trajectories")

m = train_iwcm(tv, tc, d_a=da)
rl = Rollout(8, 19, da).to(DEV)
train_rl(rl, tv, 100)

# ── Evaluate ──────────────────────────────────────────────────────────────
HORIZONS = [10, 25, 50, 100]
PHYS = slice(5, 11)  # slot channels: position(3) + velocity(3)

rows = {h: {} for h in HORIZONS}

for z0, A, Zt in ts:
    zb = z0.unsqueeze(0).to(DEV)
    Ab = A.unsqueeze(0).to(DEV)
    Zb = Zt.unsqueeze(0).to(DEV)

    # Rollout
    with torch.no_grad():
        Zr = rl.rollout(zb, Ab)

    # z0-replication solver
    Zz_init = z0.unsqueeze(0).unsqueeze(1).expand(-1, 100, -1, -1).clone().to(DEV)
    Zz = solve(m, zb, Ab, init_Z=Zz_init)

    # Warm-start solver
    Zw = solve(m, zb, Ab, init_Z=Zr, steps=100, lr=0.005)

    for h in HORIZONS:
        with torch.no_grad():
            e_true = m(zb, Ab[:, :h], Zb[:, :h]).item()
            e_roll = m(zb, Ab[:, :h], Zr[:, :h]).item()
            e_z0   = m(zb, Ab[:, :h], Zz[:, :h]).item()
            e_warm = m(zb, Ab[:, :h], Zw[:, :h]).item()

            gt    = Zb[0, :h, :, PHYS]  # (h, N, 6)
            rphys = Zr[0, :h, :, PHYS]
            zphys = Zz[0, :h, :, PHYS]
            wphys = Zw[0, :h, :, PHYS]

            rows[h].setdefault("de_roll", []).append(e_roll - e_true)
            rows[h].setdefault("de_z0",   []).append(e_z0   - e_true)
            rows[h].setdefault("de_warm", []).append(e_warm - e_true)
            rows[h].setdefault("mse_roll", []).append(F.mse_loss(rphys, gt).item())
            rows[h].setdefault("mse_z0",   []).append(F.mse_loss(zphys, gt).item())
            rows[h].setdefault("mse_warm", []).append(F.mse_loss(wphys, gt).item())

# ── Print table ───────────────────────────────────────────────────────────
print(f"\n{'H':>4} | {'Rollout MSE':>11} {'z0-rep MSE':>10} {'Warm MSE':>9} | "
      f"{'Rollout ΔE':>10} {'z0-rep ΔE':>9} {'Warm ΔE':>8}")
print("-" * 75)
for h in HORIZONS:
    r = rows[h]
    rm = np.mean(r["mse_roll"])
    zm = np.mean(r["mse_z0"])
    wm = np.mean(r["mse_warm"])
    re = np.mean(r["de_roll"])
    ze = np.mean(r["de_z0"])
    we = np.mean(r["de_warm"])
    print(f"{h:4d} | {rm:11.6f} {zm:10.6f} {wm:9.6f} | "
          f"{re:+10.3f} {ze:+9.3f} {we:+8.3f}")

# Also print as LaTeX table
print("\n% LaTeX table\n")
print("\\begin{tabular}{@{}lrrr|rrr@{}}")
print("\\toprule")
print("H & \\multicolumn{3}{c}{State MSE (pos+vel)} & \\multicolumn{3}{c}{Energy $\\Delta$} \\\\")
print("\\cmidrule(lr){2-4} \\cmidrule(lr){5-7}")
print(" & Rollout & z0-rep & Warm & Rollout & z0-rep & Warm \\\\")
print("\\midrule")
for h in HORIZONS:
    r = rows[h]
    rm = np.mean(r["mse_roll"])
    zm = np.mean(r["mse_z0"])
    wm = np.mean(r["mse_warm"])
    re = np.mean(r["de_roll"])
    ze = np.mean(r["de_z0"])
    we = np.mean(r["de_warm"])
    print(f"{h:4d} & {rm:.6f} & {zm:.6f} & {wm:.6f} & {re:+6.3f} & {ze:+6.3f} & {we:+6.3f} \\\\")
print("\\bottomrule")
print("\\end{tabular}")
