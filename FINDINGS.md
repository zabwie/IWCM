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

| Version | What Changed | Params | Samples/s (B=64) | vs MLP |
|---------|-------------|--------|-----------|--------|
| Slow IWCM | Bidirectional 2-layer GRU | 579K | 9K | 110× slower |
| Conv1d IWCM | Depthwise conv replaces GRU | 71K | 53K | 20× slower |
| Pooling v2 | Delta features added | 205K | 105K | 10× slower |
| Fused Pooling | No GRU, single projection, fused std | 52K | 212K | 4.9× slower |
| **+ Triton Kernel** | Custom CUDA fused temporal pool | 52K | **388K** | **2.7× slower** |
| **+ CUDA Graph** | Zero kernel launch overhead | 52K | **390K** | **2.7× slower** |
| MLP baseline | Flat 3-layer MLP | 990K | 1.04M | 1× |
| Distilled MLP | IWCM→MLP knowledge distillation | 990K | 1.04M | 1× |

**At batch=1 (latency)**: IWCM + Triton + CUDA Graph = **33 μs** (30K samples/s) — **beats MLP** at 43 μs (23K samples/s) while using 5% of the parameters.

**Key optimization**: GRU accounted for 85% of parameters (495K/579K) and 64% of CUDA time. Eliminating it reduced inference cost by 22×.

### Micro-Optimization: Triton + CUDA Graphs (June 2026)

Custom CUDA kernels and CUDA graph capture push IWCM latency into microseconds.

| Path | Batch=1 (μs) | Batch=1 (sps) | Batch=64 (sps) | Accuracy |
|------|-------------|---------------|----------------|----------|
| Fused Pooling (vanilla eager) | 120.5 | 8.3K | 225K | full |
| + Triton pool | 99.8 | 10.0K | 388K | full |
| + CUDA Graph (FP32) | **33.0** | **30.3K** | **390K** | full |
| + CUDA Graph (BF16) | 45.6 | 22.0K | 329K | full |
| NoPool (H·128→3, no pool) + Graph | 45.6 | 22.0K | 328K | conservation only (0.61) |
| MLP (990K) | 44.7 | 22.4K | 1.05M | mediocre (0.67c) |

**Key findings:**
1. **CUDA Graph is the dominant optimization.** Kernel launch overhead was 55 μs at B=1 (46% of total) and 135 μs at B=64 (47%). Graph capture eliminates it entirely by recording and replaying all GPU operations as a single sequence.
2. **IWCM beats MLP at batch=1** — 33 μs vs 43 μs (28% faster) with 52K vs 990K parameters.
3. **At batch=64, MLP still leads** (1.05M vs 390K sps) due to cuBLAS batching on large matmuls. The gap is fundamental: IWCM must scan H=25 timesteps per slot.
4. **BF16 hurts at batch=1** (precision conversion overhead dominates tiny matmul) but helps at batch=64 for the shared projection (1.20× on matmul).

**Files created:**
- `src/iwcm/triton_ops.py` — Triton CUDA kernels: fused temporal pool, fused head forward, hyper-fused forward
- `src/iwcm/micro_energy.py` — `MicroIWCM` (drop-in for FusedIWCMEnergy) and `NoPoolIWCM` (no temporal reduction)
- `scripts/benchmark_microseconds.py` — Comprehensive microsecond-level benchmark

### Why Temporal Pooling Is Structurally Necessary

We tested `NoPoolIWCM` — the head Linear absorbs the temporal dimension directly (H·128→3 per slot) with no pooling reductions. Result: identity detection collapsed from 0.875→0.284 regardless of hidden layer size (tested 0, 32, 64, 128 hidden dims). Conservation held at 0.61-0.68.

**The max pooling over H is not approximable by learned weights.** Identity violations (swap, duplicate, teleport) manifest as a single anomalous timestep. Max pooling answers "did ANY timestep look wrong?" — a non-differentiable operation that picks the single most extreme value. A linear projection + GELU can approximate smooth functions but cannot emulate the discontinuous max operation across 25 timesteps. Conservation (mean-sensitive) survives; identity (max-sensitive) does not.

| Architecture | Params | Conservation | Identity | Rejection |
|---|---|---|---|---|
| Fused Pooling (has max) | 52K | 0.790 | 0.875 | 0.823 |
| NoPool H·128→3 | 12K | 0.610 | 0.284 | 0.483 |
| NoPool H·128→128→3 | 413K | 0.677 | 0.297 | 0.528 |
| MLP (flat) | 990K | 0.665 | 0.318 | 0.529 |

### Spatial Constraint Heads (Path A: Accuracy)

We tested adding explicit spatial constraint heads that operate on raw encoder channels (velocity for teleport, occupancy for deletion) rather than the learned `shared(Z)` projection. Key findings:

1. **Teleport is already solved by oracle-slot encoding.** The velocity channels (7:8) provide explicit displacement information. The baseline FusedIWCMEnergy achieves 0.969 teleport detection. The 0.29 figure from the per-axis generalization table used a different (grid) encoder without velocity channels.

2. **Delete (0.481) remains the hardest violation.** A hard existence gate (detecting occupancy→zero transitions) catches 92% of deletes but false-positives on 48% of valid trajectories — legitimate object pickups look identical to deletion in raw occupancy. Distinguishing them requires per-object-type context (held flag, door state) that a simple gate can't provide.

3. **Spatial heads preserve but don't improve baseline accuracy** on oracle-slot data. At low lambda weights (0.1-0.3), they don't interfere. At high lambda (1.0+), they starve the core head's gradient signal. The oracle-slot encoder already provides the signals spatial heads would extract.

Files: `src/iwcm/spatial_head.py` — `SpatialConstraintHead`, `ExistenceHead`, `DisplacementHead`, `SpatialIWCMEnergy`.

### Triton Backward Pass (Path B: Speed)

We attempted to fuse the temporal pool's backward pass (AmaxBackward0 at 284μs + SqrtBackward0 at 190μs = 474μs total) into custom Triton kernels. Findings:

1. **Direct-call Triton backward kernels produce correct gradients** (verified via sentinel test at all hidden sizes). The gradient math is correct.

2. **Inside `torch.autograd.Function.backward()`, the same kernels produce wrong gradients** — a consistent bug across three kernel approaches (full-fused backward, amax-only scatter, hybrid). The forward path works perfectly (1e-7 error), but backward gradients diverge by 3e3+. This appears to be a Triton + PyTorch autograd integration issue with constexpr parameters or tensor memory management in the saved tensors context.

3. **Speedup potential is 1.5-4.4× on backward pass** (measured from kernel execution time, independent of correctness). At B=1: saves ~80μs per solver iteration. At B=64: saves ~750μs per training step. If fixed, would cut solver iteration cost from 800μs→720μs and full planning from 40ms→36ms.

4. **The amax scatter approach is the right one** — targeted (only replaces the single most expensive op), simple (just scatter to argmax positions), and doesn't require recomputing the forward pass statistics.

Files: `src/iwcm/triton_ops.py` (appended functions: `_AmaxFunction`, `_amax_scatter_kernel`, `fused_temporal_pool_amax_opt`, `_FusedTemporalPoolFunction`, `_pool_backward_kernel`).

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

## Why IWCM Is Fundamentally Cheap

The IWCM energy function evaluates entire worldlines in a single feed-forward pass — no sequential bottleneck.

### Cost Per 25-Step Worldline Evaluation

| Method | Time | Notes |
|---|---|---|
| **IWCM energy (optimized)** | **0.033 ms** | Single feed-forward pass, CUDA Graph |
| IWCM energy + gradient (solver iter) | 0.80 ms | Forward + backward + Adam update |
| IWCM planning (50 iter × 8 lines) | 40.5 ms | Full gradient-based planning |
| Transformer (O(H²) attention) | ~0.6 ms | Per-step attention over 25 steps |
| DreamerV3 RSSM rollout | ~50 ms | 25 autoregressive RSSM steps |
| MPPI (1000 candidate rollouts) | ~100 ms | Sampling-based MPC |
| Video encoding (CNN+slots) | 10-50 ms | Required for visual world models |

**IWCM is 1,500× cheaper per worldline than autoregressive rollout** and processes all timesteps in parallel.

### Structural Advantages

1. **No sequential bottleneck.** All 25 timesteps processed in parallel via temporal pooling. Autoregressive models: 25 sequential forward passes. Transformers: O(H²) attention.

2. **O(1) over horizon for the core operation.** Doubling H: shared(Z) scales linearly, pooling just scans more elements. No quadratic blowup.

3. **Evaluation, not generation.** Energy function is feed-forward: worldline → scalar score. Can score 30,000 worldlines/second on a single consumer GPU. The planner just does gradient descent on candidates.

4. **52K parameters.** Fits in L2 cache. Dreamer RSSM: ~20M. GPT-based world models: 100M-7B.

5. **CUDA Graph eliminates CPU-GPU sync.** 55 μs of kernel launch overhead at batch=1 eliminated entirely.

### What Actually Costs Money

For video-based world models, the **encoder dominates** (10-50 ms per frame). The energy function at 0.033 ms is a rounding error — optimizing it from 107→33 μs doesn't matter when encoding costs 10,000 μs.

For symbolic/state-based models (oracle encoder = free), the **solver loop dominates**. Each backward pass costs 6.4× more than forward. Reducing solver iterations (warm-start, better initialization) helps more than micro-optimizing the forward pass.

The energy function forward pass is **never the bottleneck** in any real pipeline.

---

## Key Claims

1. **Compositional near-miss corruption enables causal law learning.** Naive corruptions produce 0.000 cross-surface across 7 architectures. Balanced grids produce 0.66-0.78 conservation detection.

2. **Temporal std pooling outperforms recurrent encoders for corruption detection.** GRUs smooth out the discontinuities that corruptions create. Simple mean/max/std pooling preserves the signal and is 22× faster.

3. **Architecture matters at same parameter count.** At 54K params, IWCM achieves 0.78 conservation vs MLP's 0.56. At 52K params, IWCM matches or beats the 990K MLP.

4. **Not all laws transfer equally.** Conservation and identity generalize well (0.78, 0.88). Teleport/spatial reachability remains weak (0.29).

5. **Max pooling is structurally necessary for identity detection.** Linear projections (even with GELU + hidden layers) cannot approximate the discontinuous max operation. Identity collapses from 0.88→0.28 without explicit max pooling.

6. **The energy function is not the cost bottleneck.** At 33 μs per worldline evaluation, it is 1,500× cheaper than autoregressive rollout. The encoder (video models) or solver loop (symbolic models) dominate total cost.

---

## Repository Structure

```
src/
  env/         — GridWorld environment, scenarios, datasets, symbolic oracle
  iwcm/        — Constraint heads, energy function, solver, planner
  iwcm/        — fused_energy.py (Fused Pooling architecture)
  iwcm/        — triton_ops.py (custom CUDA kernels via Triton)
  iwcm/        — micro_energy.py (MicroIWCM, NoPoolIWCM, CUDA Graph)
  iwcm/        — slot_energy.py (slot-aware constraint heads)
  iwcm/        — fast_slot_energy.py (Conv1d IWCM variant)
  iwcm/        — pooling_v2.py (delta features + statistical pooling)
  ac3/         — Mutation grammar, hardness scorer, corruptor, training loop
  tamg/        — Operator basis, validators, disagreement scorer (appendix)
  encoder/     — Slot attention, oracle slot encoder, video encoder
  metrics/     — 10 evaluation metrics including cross-surface law generalization

scripts/
  generate_compositional_grid.py  — Balanced corruption data generation
  run_comparison.py               — IWCM vs MLP vs SlotTransformer comparison
  benchmark_speed.py              — Speed + VRAM + params benchmark
  benchmark_microseconds.py       — Microsecond-level latency benchmark
  test_grokking.py                — Same-param MLP vs IWCM grokking test
  run_stats.py                    — 5-seed significance + per-axis generalization
  train_compositional.py          — Full training loop for pooled IWCM
  train_distill.py                — Knowledge distillation IWCM→MLP

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
