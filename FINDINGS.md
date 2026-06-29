# IWCM — Research Findings

**Hardware**: NVIDIA RTX 3060 (12 GB) | **PyTorch**: 2.12 | **CUDA**: 12.x

---

## Core Result

Autoregressive world models drift because each predicted state becomes the next input. IWCM removes that chain by treating future states as jointly optimized latent variables under a learned trajectory energy. With z0-replication initialization, it maintains flat learned constraint energy over long horizons without requiring a rollout warm-start.

---

## Speed Optimization

| Configuration | Time | |ΔE| vs K=100 | vs K=100 |
|---|---|---|---|---|
| Exact K=100, lr=0.01 | 51.2 ms | ref | 1× |
| Exact K=20, lr=0.08 | 10.9 ms | < 0.004 | 5× |
| **Amortized solver** (76K params) | **92–108 μs** | < 0.002 vs K20 | **500×** |

The amortized solver is a per-timestep MLP trained by distilling the exact K20 solver. One forward pass, no autograd. Speedup comes from removing inference-time backward passes, not lower precision (fp16/bf16 drift |ΔE| > 2.0).

---

## Cross-Domain Continuous Control

All at H=100, z0-rep initialization, amortized solver:

| Domain | Bodies | Action dim | Amort. MSE | Amort. ΔE | Time |
|---|---|---|---|---|---|
| Cartpole swingup | 2 | 1 | 8.2e-4 | -0.83 | 100 μs |
| Cheetah run | 7 | 6 | 8.8e-3 | -4.51 | 106 μs |
| Walker walk | 7 | 6 | 4.0e-6 | -0.03 | 96 μs |

Rollout MLP achieves lower exact-trajectory MSE (regression objective), but IWCM produces consistently negative energy (constraint-satisfaction objective). Both are small in absolute normalized scale.

---

## Energy vs Physical Accuracy (Table 4)

| H | Rollout MSE | z0-rep MSE | Warm MSE | Rollout ΔE | z0-rep ΔE | Warm ΔE |
|---|---|---|---|---|---|---|
| 10 | 2e-6 | 1.7e-4 | 8.5e-5 | +0.008 | -0.108 | -0.065 |
| 25 | 5e-6 | 3.3e-4 | 1.2e-4 | +0.012 | -0.182 | -0.133 |
| 50 | 8e-6 | 5.0e-4 | 1.3e-4 | +0.026 | -0.336 | -0.277 |
| 100 | 2e-5 | 8.4e-4 | 1.6e-4 | +0.046 | -0.698 | -0.599 |

These are different objectives: rollout minimizes next-step regression, IWCM solves constraint satisfaction. The energy metric and physical accuracy are correlated but not equivalent.

---

## AC3 Compositional Corruption (Table 1)

| Violation Type | Random corruptions | Compositional | Δ |
|---|---|---|---|
| delete | 0.850 | 0.939 | +0.089 |
| duplicate | 0.783 | **0.986** | +0.203 |
| reverse | 0.598 | **0.932** | +0.334 |
| swap | 0.663 | **0.964** | +0.300 |
| teleport | 0.600 | **0.983** | +0.383 |
| transform | 0.684 | **0.910** | +0.227 |
| **Average** | **0.696** | **0.952** | **+0.256** |

---

## TAMG Pixel Pipeline (grid-world)

- TAMGSlotEncoder (345K params, no labels) → 0.996 AUROC
- Frozen ResNet-18 (11M params) → 0.63 AUROC
- Oracle slots (reference) → 0.952 AUROC
- Flat energy across 10× horizon: Δ = +0.003

---

## Drift Elimination (DM Control cartpole, H=100)

| Method | Energy at H=100 | Δ vs true |
|---|---|---|
| Random init (cold) | +52.23 | +53.19 |
| Rollout MLP | -0.50 | +0.23 |
| **z0-replication** | **-3.62** | **-2.66** |
| Warm-start (rollout → IWCM) | -3.16 | -2.20 |

Both z0-rep and warm-start produce negative energy that stays flat across H=10 to H=100 (zero slope). Cold-start and rollout drift positively with horizon.

---

## Recovery from Degraded Rollout

Even from a rollout model with 95× higher MSE (1 hidden layer, 32 units, 15 epochs), the warm-started IWCM solver recovers trajectories with energy below ground truth. The solver acts as a constraint-satisfaction filter independent of rollout quality.

---

## Key Files

| File | Purpose |
|---|---|
| `paper/IWCM.tex` | Paper source |
| `scripts/experiments/_common.py` | Shared helpers: env, data, train_iwcm, solve |
| `scripts/experiments/dm_drift.py` | Drift comparison |
| `scripts/experiments/dm_z0init.py` | z0-rep vs random vs warm-start |
| `scripts/experiments/dm_energy_mse.py` | Energy vs MSE correlation (Table 4) |
| `scripts/experiments/train_amortized_solver.py` | Amortized solver distillation |
| `scripts/experiments/solver_optimize.py` | K/lr sweep, speed benchmarks |
| `src/iwcm/fused_energy.py` | FusedIWCMEnergy (52K params) |
| `src/iwcm/micro_energy.py` | MicroIWCM (Triton-optimized forward) |
| `src/env/dm_control_wrapper.py` | DM Control environment wrapper |
| `src/env/dm_control_encoder.py` | Oracle-structured slot encoder |
