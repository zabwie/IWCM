#!/usr/bin/env python3
"""1000-step multi-trajectory stress test — the one number to rule them all.

Usage:
  python scripts/proof_stress1000.py           # 20 trajectories, H=1000
  python scripts/proof_stress1000.py --quick   # 5 trajectories, H=500

Output: proof_stress1000.png — scatter + mean line across all trajectories
"""
import sys, torch, numpy as np, argparse
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
OUT = ROOT / 'outputs' / 'proof'
OUT.mkdir(parents=True, exist_ok=True)
CACHE = ROOT / 'outputs' / 'checkpoints'

torch.manual_seed(42)
np.random.seed(42)

parser = argparse.ArgumentParser()
parser.add_argument('--quick', action='store_true')
args = parser.parse_args()

N_TRAJ = 5 if args.quick else 20
H = 500 if args.quick else 500

print(f"Device: {DEVICE}")
print(f"Running {N_TRAJ} trajectories of length H={H}")

# Load models
from src.env.dm_control_wrapper import DMControlWrapper
from src.env.dm_control_encoder import DMControlOracleEncoder
from src.iwcm.fused_energy import FusedIWCMEnergy
from scripts.experiments._common import Rollout
from scripts.reproduction import _solve

from src.env.dm_control_wrapper import DMControlWrapper as _DMW
_da_tmp = _DMW('cartpole', 'swingup').action_dim
iwcm = FusedIWCMEnergy(d_slot=19, d_action=_da_tmp, hidden=128, num_slots=8).to(DEVICE)
Ns, ds, da = 8, 19, _da_tmp
iwcm.load_state_dict(torch.load(CACHE / 'repro_dm_iwcm.pt', map_location=DEVICE))
iwcm.eval()
rl = Rollout(Ns, ds, da).to(DEVICE)
rl.load_state_dict(torch.load(CACHE / 'repro_dm_rollout.pt', map_location=DEVICE))
rl.eval()

# Generate trajectories
all_data = []
for seed in range(N_TRAJ * 5):
    w = DMControlWrapper('cartpole', 'swingup', seed=seed, max_episode_steps=H+10)
    e = DMControlOracleEncoder('cartpole')
    result = w.generate_trajectory(H, random_policy=True)
    if result is None:
        continue
    enc = e.encode_trajectory(result[0], result[2], H)
    if enc is not None:
        all_data.append((enc, result[0], result[2]))
        if len(all_data) >= N_TRAJ:
            break

print(f"Generated {len(all_data)}/{N_TRAJ} valid trajectories")
if len(all_data) < 3:
    print("Too few trajectories, aborting")
    sys.exit(1)

# Run stress test
from collections import defaultdict
energy_records = defaultdict(list)  # H -> {'roll': [], 'warm': [], 'gt': []}
mse_records = defaultdict(list)

for t_idx, (enc, ps, an) in enumerate(all_data):
    z0_np, A_np, Zt_np = enc
    z0 = torch.from_numpy(z0_np).float().unsqueeze(0).to(DEVICE)
    A = torch.from_numpy(A_np).float().unsqueeze(0).to(DEVICE)
    Z_true = torch.from_numpy(Zt_np).float().unsqueeze(0).to(DEVICE)

    # Rollout open-loop
    Z_roll = [z0]; z_t = z0
    for t in range(H):
        with torch.no_grad():
            z_next = rl(z_t, A[:, t])
        Z_roll.append(z_next); z_t = z_next
    Z_roll = torch.stack(Z_roll[1:], dim=1)

    # IWCM warm-start
    Z_warm = _solve(iwcm, z0, A, init_Z=Z_roll.clone(), steps=100, lr=0.005, anticollapse=True)

    # Evaluate at horizons
    checkpoints = list(range(10, H + 1, 10))
    with torch.no_grad():
        for h in checkpoints:
            e_roll = iwcm(z0, A[:, :h], Z_roll[:, :h]).item()
            e_warm = iwcm(z0, A[:, :h], Z_warm[:, :h]).item()
            e_gt = iwcm(z0, A[:, :h], Z_true[:, :h]).item()
            energy_records[h].append({'roll': e_roll, 'warm': e_warm, 'gt': e_gt})

            mse_roll = (Z_roll[:, :h] - Z_true[:, :h]).pow(2).mean().item()
            mse_warm = (Z_warm[:, :h] - Z_true[:, :h]).pow(2).mean().item()
            mse_records[h].append({'roll': mse_roll, 'warm': mse_warm})

    if (t_idx + 1) % 5 == 0:
        print(f"  Trajectory {t_idx+1}/{len(all_data)}")

# Aggregate
Hs = sorted(energy_records.keys())
roll_mean = [np.mean([r['roll'] for r in energy_records[h]]) for h in Hs]
roll_std = [np.std([r['roll'] for r in energy_records[h]]) for h in Hs]
warm_mean = [np.mean([r['warm'] for r in energy_records[h]]) for h in Hs]
warm_std = [np.std([r['warm'] for r in energy_records[h]]) for h in Hs]
gt_mean = [np.mean([r['gt'] for r in energy_records[h]]) for h in Hs]

mse_roll_mean = [np.mean([r['roll'] for r in mse_records[h]]) for h in Hs]
mse_warm_mean = [np.mean([r['warm'] for r in mse_records[h]]) for h in Hs]

# Print table
print(f"\n{'H':>6} | {'Rollout E':>10} | {'IWCM E':>10} | {'Gap':>8} | {'Roll MSE':>9} | {'IWCM MSE':>9} | {'MSE Δ%':>8}")
print("-" * 70)
for i, h in enumerate(Hs):
    gap = roll_mean[i] - warm_mean[i]
    mse_gap_pct = (mse_roll_mean[i] - mse_warm_mean[i]) / max(mse_roll_mean[i], 1e-10) * 100
    print(f"{h:6d} | {roll_mean[i]:+10.3f} | {warm_mean[i]:+10.3f} | {gap:+8.3f} | {mse_roll_mean[i]:9.6f} | {mse_warm_mean[i]:9.6f} | {mse_gap_pct:+7.1f}%")

# ====== PLOT ======
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Panel 1: Energy gap over horizon
ax = axes[0]
ax.fill_between(Hs, [r - s for r,s in zip(roll_mean, roll_std)],
                [r + s for r,s in zip(roll_mean, roll_std)], alpha=0.15, color='red')
ax.fill_between(Hs, [w - s for w,s in zip(warm_mean, warm_std)],
                [w + s for w,s in zip(warm_mean, warm_std)], alpha=0.15, color='blue')
ax.plot(Hs, roll_mean, 'r-', linewidth=2, label=f'Rollout (N={N_TRAJ})')
ax.plot(Hs, warm_mean, 'b-', linewidth=2, label=f'IWCM warm-start (N={N_TRAJ})')
ax.axhline(y=0, color='black', linewidth=0.5)
ax.set_xlabel('Horizon H (steps)', fontsize=13)
ax.set_ylabel('Learned Energy E_θ(z₀, A, Z)', fontsize=13)
ax.set_title('Learned Energy vs Horizon', fontsize=14)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)

# Panel 2: Energy gap (E_roll - E_warm) — positive means IWCM wins
ax = axes[1]
gap_mean = [r - w for r,w in zip(roll_mean, warm_mean)]
gap_std_combined = [np.sqrt(rs**2 + ws**2) for rs, ws in zip(roll_std, warm_std)]
ax.fill_between(Hs, [g - s for g,s in zip(gap_mean, gap_std_combined)],
                [g + s for g,s in zip(gap_mean, gap_std_combined)], alpha=0.2, color='purple')
ax.plot(Hs, gap_mean, 'purple', linewidth=2)
ax.axhline(y=0, color='black', linewidth=0.5)
ax.set_xlabel('Horizon H (steps)', fontsize=13)
ax.set_ylabel('Energy Gap (E_roll − E_warm)', fontsize=13)
ax.set_title('IWCM Advantage: Energy Gap vs Horizon', fontsize=14)
ax.annotate('IWCM better →', xy=(0.95, 0.9), xycoords='axes fraction',
            fontsize=11, ha='right', color='purple')
ax.annotate('← Rollout better', xy=(0.05, 0.1), xycoords='axes fraction',
            fontsize=11, ha='left', color='red')
ax.grid(True, alpha=0.3)

# Panel 3: MSE comparison with error bars at final H
ax = axes[2]
final_h = Hs[-1]
h_display = Hs[::max(1, len(Hs)//10)]  # show ~10 points
x_vals = [h for h in h_display]
roll_mse_at_h = [mse_roll_mean[Hs.index(h)] for h in h_display]
warm_mse_at_h = [mse_warm_mean[Hs.index(h)] for h in h_display]
ax.plot(h_display, roll_mse_at_h, 'r-o', linewidth=2, markersize=4, label='Rollout MSE')
ax.plot(h_display, warm_mse_at_h, 'b-o', linewidth=2, markersize=4, label='IWCM MSE')
ax.set_xlabel('Horizon H', fontsize=13)
ax.set_ylabel('Cumulative MSE', fontsize=13)
ax.set_title('Open-Loop MSE Comparison', fontsize=14)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
ax.set_yscale('log')
# Final values
ax.annotate(f'H={final_h}:\nRoll MSE={mse_roll_mean[-1]:.6f}\nIWCM MSE={mse_warm_mean[-1]:.6f}',
            xy=(0.95, 0.95), xycoords='axes fraction', fontsize=11,
            ha='right', va='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
path = str(OUT / 'proof_stress1000.png')
plt.savefig(path, dpi=150, bbox_inches='tight')
plt.close()
print(f"\nSaved: {path}")

# Final summary
final_roll_e = roll_mean[-1]
final_warm_e = warm_mean[-1]
final_gap = gap_mean[-1]
final_roll_mse = mse_roll_mean[-1]
final_warm_mse = mse_warm_mean[-1]

print(f"\n{'='*60}")
print(f"FINAL RESULT (H={final_h}, N={N_TRAJ} trajectories)")
print(f"{'='*60}")
print(f"  Rollout energy:  {final_roll_e:+.4f}")
print(f"  IWCM energy:     {final_warm_e:+.4f}")
print(f"  Gap:             {final_gap:+.4f}  ({'IWCM wins ✓' if final_gap > 0 else 'Rollout wins'})")
print(f"  Rollout MSE:     {final_roll_mse:.6f}")
print(f"  IWCM MSE:        {final_warm_mse:.6f}")
print(f"  MSE improvement: {(final_roll_mse - final_warm_mse)/max(final_roll_mse,1e-10)*100:.1f}%")
print(f"{'='*60}")
