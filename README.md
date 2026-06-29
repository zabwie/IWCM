# IWCM — Implicit Worldline Constraint Models

**Joint trajectory energy optimization without autoregressive rollout.**

Learned world models based on autoregressive rollout suffer from a structural failure: prediction errors compound over time because each predicted state feeds into the next prediction. IWCM reframes world modeling as full-trajectory constraint satisfaction — the energy function scores entire future trajectories jointly, and the solver finds a valid worldline by gradient descent rather than step-by-step generation.

Key results — [full paper](paper/IWCM.pdf):

| Claim | Evidence |
|---|---|
| Autoregressive chain drift removed by construction | Energy flat across H=10–100 (slope 0.0015), rollout baseline drifts +53 |
| No rollout model needed | z0-replication initialization achieves lower energy than warm-start |
| Exact solver → amortized 92 μs | 76K-parameter feed-forward network matches K20 energy within \|ΔE\| < 0.002 |
| Cross-domain (3 continuous-control domains) | Cartpole, cheetah-run, walker-walk — all ~100 μs |
| AC3 curriculum learns invariants | 0.952 AUROC vs 0.696 random corruptions |
| Pixel pipeline (grid-world + MuJoCo) | TAMGSlotEncoder (345K, no labels): 0.996 AUROC grid-world, 0.971–0.978 on 3 MuJoCo domains |
| Energy vs trajectory accuracy | Both metrics reported; they optimize different objectives |

## Project Structure

```
ICWM/
├── paper/                    # LaTeX source and PDF
├── src/
│   ├── env/                  # Environments and encoders
│   │   ├── grid_world.py     # Grid world environment
│   │   ├── dm_control_wrapper.py
│   │   ├── dm_control_encoder.py
│   │   └── oracle_slot_encoder.py
│   ├── iwcm/
│   │   ├── fused_energy.py   # FusedIWCMEnergy (52K params)
│   │   ├── solver.py         # Gradient descent solver
│   │   ├── micro_energy.py   # Latency-optimized forward (Triton)
│   │   ├── triton_ops.py     # Custom Triton CUDA kernels
│   │   ├── energy.py         # 5-head energy function
│   │   ├── slot_energy.py    # Slot-aware 5-head energy
│   │   └── planner.py        # MAP inference planner
│   └── env/__init__.py
├── scripts/
│   ├── experiments/          # Paper experiment scripts
│   │   ├── _common.py        # Shared helpers (env, data, train_iwcm, solve)
│   │   ├── dm_drift.py       # Drift comparison
│   │   ├── dm_z0init.py      # z0-replication vs random vs warm-start
│   │   ├── dm_recovery.py    # Degraded rollout recovery
│   │   ├── dm_energy_mse.py  # Energy vs physical MSE correlation (Table 4)
│   │   ├── solver_optimize.py # K/lr sweep, speed benchmarks
│   │   ├── solver_profile.py # Step-by-step profiling
│   │   └── train_amortized_solver.py  # Amortized solver distillation
│   ├── train/                # Training scripts
│   ├── bench/                # Microsecond-level benchmarks
│   ├── diagnostics/          # Diagnostic experiments
│   └── data/                 # Data generation
└── pyproject.toml
```

## Quick Start

### Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

Requires PyTorch ≥ 2.1 and CUDA (tested on RTX 3060).

### Run DM Control Drift Experiment

```bash
python scripts/experiments/dm_drift.py
```

### Train Amortized Solver

```bash
python scripts/experiments/train_amortized_solver.py
```

### Benchmark Solver Speed

```bash
python scripts/experiments/solver_optimize.py
```

### Train TAMG Pixel Encoder (Continuous Control)

```bash
python scripts/train_tamg_dm_control.py
```

Trains TAMGSlotEncoder + FusedIWCMEnergy on rendered MuJoCo frames. Self-supervised: velocity MSE, contrastive identity, clustering type, reconstruction + IWCM margin loss. ~45 min per domain on RTX 3060.

All experiment scripts are self-contained — they generate data, train models, and produce results in a single run.

## Solver Configuration

| Configuration | Time | Energy accuracy | Use case |
|---|---|---|---|
| K=100, lr=0.01 | 51.2 ms | Reference | Paper baseline |
| K=20, lr=0.08 | 10.9 ms | \|ΔE\| < 0.004 | Practical exact solver |
| Amortized (76K params) | 92–108 μs | \|ΔE\| < 0.002 | Fast inference |

The amortized solver is trained by distilling the exact K20 solver into a per-timestep MLP. See `scripts/experiments/train_amortized_solver.py`.

## Domains

### Continuous Control (MuJoCo via DM Control)

| Domain | Bodies | Action dim | Oracle-slot results | Pixel pipeline (TAMG) |
|---|---|---|---|---|
| Cartpole swingup | 2 | 1 | Full results | 0.978 AUROC |
| Cheetah run | 7 | 6 | Full results | 0.971 AUROC |
| Walker walk | 7 | 6 | Full results | 0.978 AUROC |

### Grid World

Symbolic oracle and pixel-only variants supported. TAMGSlotEncoder achieves 0.996 AUROC from pixel input (345K params, no labels).

## Citation

```bibtex
@article{iwcm2026,
  title={Implicit Worldline Constraint Models:
         Joint Trajectory Energy Optimization Without Autoregressive Rollout},
  author={P{\'e}rez Mu{\~n}iz, Zabdiel},
  year={2026}
}
```
