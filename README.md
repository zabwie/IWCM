# IWCM — Implicit Worldline Constraint Models

**World Models as Constraint Solvers**

Implementation of the IWCM/AC3/TAMG framework for self-supervised causal law discovery in world models.

## Paper

*World Models as Constraint Solvers: Implicit Worldline Constraint Models, Adversarial Causal Corruption Curricula, and Self-Supervised Causal Law Discovery*

Located at: `paper/IWCM.tex` and `paper/IWCM.pdf`

## Architecture

The framework has three complementary components:

1. **IWCM** — Energy-based world model over complete latent worldlines. Planning as joint constraint satisfaction, not autoregressive rollout.
2. **AC3** — Adversarial Causal Corruption Curriculum. Co-evolutionary training with learned corruptor generating near-miss worlds.
3. **TAMG** — Tangent-Space Adversarial Mutation Grammar with Validator Committee. Self-supervised extension eliminating symbolic oracle dependency.

## Project Structure

```
ICWM/
├── paper/                    # LaTeX source, PDF, and figures
├── src/                      # Source code
│   ├── env/                  # Environments and encoders
│   │   ├── grid_world.py     # Grid world environment
│   │   ├── objects.py        # Key/Door/Box/Occluder definitions
│   │   ├── actions.py        # Action definitions
│   │   ├── renderer.py       # Video frame renderer
│   │   ├── scenarios.py      # Predefined scenarios
│   │   ├── data.py           # PyTorch datasets
│   │   ├── symbolic_state.py # Oracle state access / symbolic encoding
│   │   ├── oracle_slot_encoder.py  # Oracle slot encoder (19-dim per slot)
│   │   ├── dm_control_wrapper.py   # DM Control environment wrapper
│   │   └── dm_control_encoder.py   # DM Control oracle slot encoder
│   ├── iwcm/                 # IWCM core
│   │   ├── energy.py         # 5-head IWCM energy function E_θ
│   │   ├── fused_energy.py   # Fused pooling IWCM (52K params, main architecture)
│   │   ├── slot_energy.py    # Slot-aware 5-head energy function
│   │   ├── solver.py         # Gradient descent worldline solver
│   │   ├── planner.py        # MAP inference planner
│   │   ├── refinement.py     # Learned refinement operator Φ_θ
│   │   ├── model.py          # Full IWCM wrapper
│   │   ├── micro_energy.py   # Latency-optimized MicroIWCM (Triton + CUDA Graph)
│   │   ├── triton_ops.py     # Custom Triton CUDA kernels
│   │   ├── constraints/      # 5 constraint head implementations
│   │   └── variants/         # Architecture variants (conv1d, pooling_v2, spatial)
│   ├── ac3/                  # AC3 adversarial training
│   │   ├── mutations/        # 7 symbolic mutation types (grammar.py)
│   │   ├── corruptor.py      # Learned corruptor C_φ
│   │   ├── hardness.py       # Hardness scorer + curriculum
│   │   ├── oracle.py         # Symbolic constraint oracle
│   │   └── trainer.py        # AC3 training loop (Alg. 1)
│   ├── tamg/                 # TAMG self-supervised
│   │   ├── operators.py      # Learned operator basis
│   │   ├── slot_encoder.py   # TAMGSlotEncoder (pixels → 19-dim slots)
│   │   ├── mutations/        # 6 continuous mutation families
│   │   ├── corruptor.py      # Manifold-preserving corruptor
│   │   ├── validators/       # 8-member validator committee
│   │   ├── disagreement.py   # Disagreement score D(τ')
│   │   └── trainer.py        # TAMG training loop (Alg. 2)
│   ├── encoder/              # Video encoder (Exp 2)
│   │   ├── video_encoder.py  # CNN + slot attention
│   │   ├── slot_attention.py # Iterative slot attention (with spatial anchoring)
│   │   ├── decoder.py        # Spatial broadcast decoder
│   │   ├── representation.py # Content/pose/hidden decomposition
│   │   ├── slot_permanence.py # Temporal slot identity tracking
│   │   ├── slot_transition.py # Predicted next-frame slot init
│   │   ├── slot_structure.py # Weak oracle supervision heads
│   │   └── spatial_anchor.py # Spatial anchoring utilities
│   ├── metrics/              # 9 evaluation metrics
│   │   └── evaluation.py
│   ├── utils/                # Utilities
│   │   ├── config.py         # Hydra/OmegaConf config
│   │   ├── seed.py           # Reproducibility
│   │   ├── logging.py        # Wandb + TensorBoard
│   │   ├── tensors.py        # Worldline slab ops
│   │   └── base.py           # Base model class
│   ├── tamg_simple.py        # Simplified TAMG training harness
│   └── pixel_slots.py        # Pixel-based slot utilities
├── configs/                  # YAML configuration
│   ├── default.yaml
│   ├── exp1/                 # Experiment 1 configs
│   └── exp2/                 # Experiment 2 configs
├── experiments/              # Experiment runner stubs
│   ├── exp1_symbolic/
│   ├── exp2_video/
│   └── analysis/
├── scripts/                  # CLI scripts (organized by function)
│   ├── experiments/          # Paper experiment scripts (drift, recovery, etc.)
│   ├── train/                # Training scripts
│   ├── data/                 # Data generation scripts
│   ├── diagnostics/          # Diagnostic and ablation scripts
│   ├── bench/                # Benchmark scripts
│   ├── stage2/               # Stage 2 (slot permanence) scripts
│   ├── exp1/                 # Experiment 1 analysis scripts
│   ├── dm_control/           # DM Control training scripts
│   ├── run_experiment.py     # Unified experiment runner
│   ├── evaluate.py           # Model evaluation
│   ├── generate_data.py      # Data generation
│   └── ...                   # Additional analysis scripts
├── tests/                    # Test suites
├── data/                     # Generated trajectory datasets
├── outputs/                  # Checkpoints, logs
├── notebooks/                # Jupyter notebooks (empty)
├── FINDINGS.md               # Research findings and results
├── pyproject.toml
└── README.md
```

## Quick Start

### Installation

```bash
pip install -e .
# For dev tools:
pip install -e ".[dev]"
```

### Generate Training Data

```bash
# Generate trajectories for all scenarios
python scripts/generate_data.py --all --horizon 25 --num 10000 --seed 42

# Generate counterfactual pairs
python scripts/generate_data.py --counterfactuals --horizon 25 --num 1000
```

### Run Experiments

```bash
# Experiment 1: Symbolic grid world with AC3
python scripts/run_experiment.py --exp exp1 --model c --horizon 25 --epochs 100

# Experiment 2: Video environment with TAMG
python scripts/run_experiment.py --exp exp2 --model d --horizon 25 --epochs 150
```

### Evaluate a Model

```bash
python scripts/evaluate.py --checkpoint outputs/checkpoints/model.pt --exp exp1
```

## Experiments

### Experiment 1: Symbolic Grid World
- **Model A**: Baseline next-state predictor
- **Model B**: IWCM + random corruptions (no adversary)
- **Model C**: IWCM + AC3 with symbolic oracle

### Experiment 2: Video Environment
- **Model A**: Random latent noise baseline
- **Model B**: Hand-coded symbolic (upper bound)
- **Model C**: TAMG with Validator Committee
- **Model D**: TAMG + structured validator disagreement

## Evaluation Metrics

1. Constraint violation rate over H ∈ {10, 25, 50, 100}
2. Object identity preservation
3. Conservation violation detection (held-out)
4. Valid/invalid future classification
5. Repair accuracy on corrupted worldlines
6. Counterfactual locality accuracy
7. Splice detection (Δt > 20)
8. Planning success rate
9. **Cross-surface law generalization** (primary metric)

## Requirements

- Python ≥ 3.10
- PyTorch ≥ 2.1
- CUDA (recommended for training)
- See `pyproject.toml` for full dependency list

## Reproducibility

All experiments use fixed seeds. Configurations are in `configs/`. Model checkpoints and logs are saved to `outputs/`.

```bash
export IWCM_SEED=42
python scripts/run_experiment.py --exp exp1 --model c --seed 42
```

## Citation

```bibtex
@article{iwcm2026,
  title={World Models as Constraint Solvers: Implicit Worldline Constraint Models,
         Adversarial Causal Corruption Curricula, and Self-Supervised Causal Law Discovery},
  author={Anonymous},
  year={2026}
}
```
