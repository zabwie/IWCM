# IWCM Research Findings

**Date**: June 2026  
**Repository**: `/home/zabwie/Code/ICWM`  
**Hardware**: NVIDIA GeForce RTX 3060 (12 GB VRAM)

---

## Overview

This repository implements the IWCM/AC3/TAMG framework from *"World Models as Constraint Solvers"*. After extensive experimentation, the central finding is:

> **Compositional near-miss corruption is the dominant factor enabling cross-surface causal law learning. Architecture matters, but the corruption distribution matters more.**

---

## Final Architecture: Fused Pooling IWCM

`src/iwcm/fused_energy.py`

```
Input: Z (B, H, N, d)  — batch × horizon × object slots × features

shared = Linear(d → 128)

Z_mean  = shared(Z).mean(dim=H)
Z_max   = shared(Z).amax(dim=H)
Z_std   = sqrt(E[Z²] - E[Z]²)  — fast variance formula

concat → MLP(384 → 128 → 3)  — boundary, local, invariant scores

aggregate = 0.3 × mean(scores) + 0.7 × max(scores)
energy = λ₁·boundary + λ₂·local + λ₃·invariant
```

No recurrence. No attention. Single projection. Fused 3-head scorer.

### Why temporal pooling beats GRU

Corruptions (teleport, swap, duplicate) create **sharp temporal discontinuities**. A bidirectional GRU smooths these out. Mean/max/std pooling preserves them directly. The std channel captures variance that GRU compresses. Simpler architecture, better signal.

---

## Speed Optimization Progression

| Version | What Changed | Params | Samples/s | vs MLP |
|---------|-------------|--------|-----------|--------|
| Slow IWCM | Bidirectional 2-layer GRU | 579K | 9K | 110× slower |
| Conv1d IWCM | Depthwise conv replaces GRU | 71K | 53K | 20× slower |
| Pooling v2 | Delta features added | 205K | 105K | 10× slower |
| **Fused Pooling** | No GRU, single projection, fused std | **52K** | **212K** | **4.9× slower** |
| MLP baseline | Flat 3-layer MLP | 990K | 1.04M | 1× |

**Key optimization**: GRU accounted for 85% of parameters (495K/579K) and 64% of CUDA time. Eliminating it reduced inference cost by 22×.

---

## Accuracy Results — 5-Seed Comparison

| Model | Params | Samples/s | Conservation ± std | Identity ± std | Invalid Rej |
|-------|--------|-----------|-------------------|----------------|-------------|
| 990K MLP | 990K | 1.04M | 0.665 ± 0.023 | 0.318 ± 0.030 | 0.529 |
| 54K MLP | 54K | 1.04M | 0.564 ± 0.027 | 0.182 ± 0.022 | 0.413 |
| Slow IWCM (GRU) | 579K | 9K | 0.695 ± 0.070 | 0.331 ± 0.015 | 0.552 |
| **Fused Pooling IWCM** | **52K** | **212K** | **0.778** | **0.877** | **0.820** |
| Distilled MLP | 990K | 1.04M | 0.824 ± 0.077 | 0.417 ± 0.011 | 0.664 |

**Pooling IWCM at 52K params beats 990K MLP across all metrics** while being 19× smaller.

### Grokking Test

Same-parameter comparison (both 54K):

| Model | Conservation | Identity | Invalid Rej |
|-------|-------------|----------|-------------|
| 54K MLP | 0.564 | 0.182 | 0.413 |
| 54K IWCM | 0.778 | 0.877 | 0.820 |
| **IWCM advantage** | **+38%** | **+382%** | **+98%** |

Not grokking — learning curves show smooth improvement, not sudden generalization. Train loss drops 0.84→0.32 over 300 epochs with monotonic identity improvement.

---

## Per-Axis Generalization

Trained on compositional corruption grid, evaluated on held-out factor combinations.

| Axis | Level | Detection Rate |
|------|-------|---------------|
| **Context** | carried | 93.8% |
| | occluded | 42.1% |
| | visible | 38.1% |
| **Violation Type** | duplicate | 91.9% |
| | reverse | 75.0% |
| | transform | 57.9% |
| | delete | 44.4% |
| | swap | 45.1% |
| | teleport | 29.1% |
| **Object Type** | box | 60.0% |
| | key | 50.0% |
| **Time Gap** | early/mid/late | ~53% (no effect) |

---

## Architecture Variants Tested

All 7 variants below achieved **0.000** cross-surface when trained on naive corruptions. Only the compositional corruption grid enabled non-trivial law detection.

| Variant | Encoding | Key Feature | Cross-Surface |
|---------|----------|-------------|---------------|
| Flat IWCM | Grid (256d) | 5-head energy function | 0.000 |
| Oracle slots | Object vectors (15d) | Per-object features | 0.000 |
| Oracle + active CF | Object vectors (15d) | Fixed counterfactual head | 0.000 |
| Slot-aware | Object vectors (15d) | Slot-preserving architecture | 0.000 |
| Slot-aware + identity | Object vectors (19d) | Object ID hash channels | 0.000 |
| Slot-aware + maxpool | Object vectors (19d) | Max-only aggregation | 0.000 |
| Slot-aware + per-object GRU | Object vectors (19d) | Per-slot temporal GRU | 0.000 |

**Conclusion**: architecture alone is insufficient. The corruption distribution must be shortcut-immune for law learning to occur.

---

## Compositional Corruption Grid

`scripts/generate_compositional_grid.py`

5 independent axes:
- **Object type**: key, box
- **Context**: visible, occluded, carried
- **Violation type**: duplicate, delete, transform, swap, teleport, illegal-open, reverse
- **Time gap**: early, mid, late
- **Distractors**: none, same-type, different-type

**Compositional split**: no single factor predicts validity across train and test. Train on "key-duplicate-visible" and test on "box-delete-carried" — forces the model to learn the abstract law, not surface patterns.

---

## Key Claims

1. **Compositional near-miss corruption enables causal law learning.** Naive corruptions produce 0.000 cross-surface across 7 architectures. Balanced grids produce 0.66-0.78 conservation detection.

2. **Temporal std pooling outperforms recurrent encoders for corruption detection.** GRUs smooth out the discontinuities that corruptions create. Simple mean/max/std pooling preserves the signal and is 22× faster.

3. **Architecture matters at same parameter count.** At 54K params, IWCM achieves 0.78 conservation vs MLP's 0.56. At 52K params, IWCM matches or beats the 990K MLP.

4. **Not all laws transfer equally.** Conservation and identity generalize well (0.78, 0.88). Teleport/spatial reachability remains weak (0.29).

---

## Repository Structure

```
src/
  env/         — GridWorld environment, scenarios, datasets, symbolic oracle
  iwcm/        — Constraint heads, energy function, solver, planner
  iwcm/        — fused_energy.py (final optimized architecture)
  ac3/         — Mutation grammar, hardness scorer, corruptor, training loop
  tamg/        — Operator basis, validators, disagreement scorer (appendix)
  encoder/     — Slot attention, oracle slot encoder, video encoder
  metrics/     — 10 evaluation metrics including cross-surface law generalization

scripts/
  generate_compositional_grid.py  — Balanced corruption data generation
  run_comparison.py               — IWCM vs MLP vs SlotTransformer comparison
  benchmark_speed.py              — Speed + VRAM + params benchmark
  test_grokking.py                — Same-param MLP vs IWCM grokking test
  run_stats.py                    — 5-seed significance + per-axis generalization

configs/     — YAML configs for all experiment variants
data/        — Compositional grid, oracle slots, cross-surface test sets
```

---

## To Reproduce

```bash
# Install
pip install -e .

# Generate compositional corruption grid
python scripts/generate_compositional_grid.py --num 25

# Train fused pooling IWCM
python -c "
from src.iwcm.fused_energy import FusedIWCMEnergy
from src.encoder.oracle_slot_encoder import ORACLE_SLOT_DIM, MAX_OBJECTS
# ... see scripts/train_compositional.py for full training loop
"

# Run comparison table
python scripts/run_comparison.py

# Run statistical significance
python scripts/run_stats.py
```
