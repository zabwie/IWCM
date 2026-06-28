#!/usr/bin/env python3
"""Generate Figure 1 (drift divergence) and Figure 2 (recovery) for the paper.
   Saves as paper/fig1_drift.pdf and paper/fig2_recovery.pdf."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os

DPI = 150
WIDTH = 6.5   # inches (column width)
OUT = os.path.join(os.path.dirname(__file__), '..', '..', 'paper')

# ── Figure 1: Drift divergence ──────────────────────────────
# Data from dm_drift.py (cold) and dm_z0init.py (z0rep, warm)
steps = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

# Cold start: Δ vs true (starts at +45, rises to +53)
cold_d = [45.04, 48.88, 50.50, 51.51, 52.16, 52.67, 52.87, 53.05, 53.21, 53.19]

# z0rep: Δ vs true (flat at -2.66)
z0rep_d = [-1.29, -1.58, -1.79, -1.97, -2.09, -2.20, -2.37, -2.48, -2.56, -2.66]

# Warm-start: Δ vs true (flat at -2.20)
warm_d = [-0.90, -1.18, -1.39, -1.56, -1.67, -1.77, -1.92, -2.03, -2.10, -2.20]

fig, ax = plt.subplots(1, 1, figsize=(WIDTH, WIDTH * 0.5))

ax.plot(steps, cold_d, 'r-', linewidth=2.0, label='Cold start (random init)')
ax.plot(steps, z0rep_d, 'b-', linewidth=2.0, label='z0-replication init')
ax.plot(steps, warm_d, 'g-', linewidth=2.0, label='Warm-start (rollout init)')
ax.axhline(y=0, color='black', linestyle='--', linewidth=1.0, alpha=0.5, label='Ground truth')

ax.set_xlabel('Step', fontsize=12)
ax.set_ylabel('Energy gap Δ vs ground truth', fontsize=12)
ax.set_xlim(0, 100)
ax.set_ylim(-5, 60)
ax.set_xticks([0, 20, 40, 60, 80, 100])
ax.set_yticks([-4, 0, 10, 20, 30, 40, 50, 60])
ax.legend(loc='upper left', fontsize=9, framealpha=0.9)
ax.grid(True, alpha=0.3)
ax.set_title('Drift divergence at H=100 (DM Control cartpole)', fontsize=13)

plt.tight_layout()
plt.savefig(os.path.join(OUT, 'fig1_drift.pdf'), dpi=DPI)
plt.close()
print("fig1_drift.pdf saved")

# ── Figure 2: Recovery from degraded rollout ────────────────
# Data from dm_recovery.py
# Ground truth energy decays from -1.43 to -0.69
gt_e = [-1.43, -1.38, -1.33, -1.29, -1.24, -1.18, -1.07, -0.97, -0.84, -0.69]

# Degraded rollout: flat at -1.44
roll_e = [-1.47, -1.46, -1.45, -1.45, -1.45, -1.45, -1.44, -1.44, -1.44, -1.44]

# Warm-start IWCM: flat at -1.53
warm_e = [-1.54, -1.53, -1.53, -1.53, -1.53, -1.53, -1.53, -1.53, -1.53, -1.53]

fig2, ax2 = plt.subplots(1, 1, figsize=(WIDTH, WIDTH * 0.45))

ax2.plot(steps, roll_e, 'r-', linewidth=2.0, label='Degraded rollout')
ax2.plot(steps, warm_e, 'b-', linewidth=2.0, label='Warm-start (IWCM)')
ax2.plot(steps, gt_e, 'k--', linewidth=1.5, alpha=0.6, label='Ground truth')

ax2.set_xlabel('Step', fontsize=12)
ax2.set_ylabel('Energy $E_θ$', fontsize=12)
ax2.set_xlim(0, 100)
ax2.set_ylim(-1.7, -0.4)
ax2.set_xticks([0, 20, 40, 60, 80, 100])
ax2.legend(loc='lower left', fontsize=10, framealpha=0.9)
ax2.grid(True, alpha=0.3)
ax2.set_title('Recovery from degraded rollout (DM Control cartpole)', fontsize=13)

plt.tight_layout()
plt.savefig(os.path.join(OUT, 'fig2_recovery.pdf'), dpi=DPI)
plt.close()
print("fig2_recovery.pdf saved")
