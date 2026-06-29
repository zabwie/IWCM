"""Generate flat energy pixel plot (Figure 4)."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# Data from test_tamg_drift.py run
horizons = [10, 25, 50, 100]
tamg_energy = [-1.695, -1.694, -1.693, -1.692]
tamg_std = [0.001, 0.001, 0.001, 0.002]
rand_energy = [12.625, 14.265, 15.313, 16.165]
rand_std = [0.378, 0.320, 0.211, 0.159]

fig, ax = plt.subplots(figsize=(6, 3.5))

ax.errorbar(horizons, tamg_energy, yerr=tamg_std, fmt='o-', color='#2ecc71',
            linewidth=2, capsize=4, markersize=6, label='TAMG slots (pixels)')
ax.errorbar(horizons, rand_energy, yerr=rand_std, fmt='s--', color='#e74c3c',
            linewidth=2, capsize=4, markersize=6, label='Random initialization')

ax.axhline(y=-1.694, color='#2ecc71', linestyle=':', alpha=0.3, linewidth=1)
ax.annotate(f'Slope vs log H = 0.0015', xy=(0.5, 0.08),
            xycoords='axes fraction', ha='center', fontsize=9,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgray', alpha=0.3))

ax.set_xlabel('Horizon H', fontsize=11)
ax.set_ylabel('Energy E(Z)', fontsize=11)
ax.set_title('Flat energy across horizon: TAMG pixel slots', fontsize=12)
ax.set_xscale('log')
ax.set_xticks(horizons)
ax.set_xticklabels([str(h) for h in horizons])
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
ax.set_ylim(-3, 18)

plt.tight_layout()
plt.savefig('paper/fig3_tamg_drift.pdf', bbox_inches='tight', dpi=150)
print("Saved paper/fig3_tamg_drift.pdf")
