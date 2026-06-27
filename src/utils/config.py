"""Configuration management for IWCM experiments.

Uses OmegaConf dataclasses (Hydra-compatible). All configs are
structured dataclasses with type validation.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Tuple
from pathlib import Path
import yaml

try:
    from omegaconf import OmegaConf, DictConfig
    HAS_OMEGACONF = True
except ImportError:
    HAS_OMEGACONF = False


# ═══════════════════════════════════════════════════════════
# Config Dataclasses
# ═══════════════════════════════════════════════════════════

@dataclass
class EnvConfig:
    """Grid world environment configuration."""
    grid_size: int = 8
    max_objects: int = 5
    num_keys: int = 1
    num_doors: int = 1
    num_boxes: int = 2
    num_occluders: int = 1
    max_steps_per_episode: int = 200


@dataclass
class IWCMConfig:
    """IWCM model configuration (Section 4)."""
    # Latent dimensions
    latent_dim: int = 64
    num_slots: int = 6
    hidden_dim: int = 256

    # Transformer architecture
    num_heads: int = 4
    num_layers: int = 2
    dropout: float = 0.1

    # Constraint head weights (λ₁–λ₅)
    lambda_boundary: float = 1.0
    lambda_local: float = 1.0
    lambda_invariant: float = 1.5
    lambda_effect: float = 1.0
    lambda_counterfactual: float = 0.5

    # Solver configuration
    solver_steps: int = 20
    solver_lr: float = 0.01
    solver_momentum: float = 0.9
    use_learned_refinement: bool = True

    # Acceptance scoring
    accept_temperature: float = 1.0


@dataclass
class AC3Config:
    """AC3 training configuration (Section 5)."""
    # Corruptor
    num_mutations_per_sample: int = 4
    mutation_grammar_weights: List[float] = field(
        default_factory=lambda: [1.0, 1.0, 1.0, 1.0, 1.0, 0.5, 0.5]
    )  # identity, conservation, locality, splice, counterfactual, temporal, occlusion

    # Curriculum
    accept_low: float = 0.4
    accept_high: float = 0.7
    top_k_hard: int = 8
    curriculum_warmup: int = 1000

    # Corruptor loss weights
    lambda_invalid: float = 1.0
    lambda_accept: float = 0.5
    lambda_minimal: float = 0.3
    lambda_diversity: float = 0.2

    # World model loss weights
    lambda_valid_energy: float = 1.0
    lambda_invalid_energy: float = 1.0
    lambda_repair: float = 0.5

    # Oracle
    use_symbolic_oracle: bool = True


@dataclass
class EncoderConfig:
    """Video encoder configuration (Section 6, Experiment 2)."""
    cnn_channels: List[int] = field(default_factory=lambda: [32, 64, 64, 128])
    frame_size: int = 64
    num_slots: int = 6
    slot_dim: int = 64
    slot_iters: int = 3
    content_dim: int = 32
    pose_dim: int = 16
    hidden_dim: int = 16  # occlusion/confidence


@dataclass
class TAMGConfig:
    """TAMG configuration (Section 6)."""
    # Operator basis
    num_operators: int = 16
    operator_dim: int = 32
    operator_rank: int = 8  # low-rank adapter rank
    num_operator_clusters: int = 16

    # Mutation families (which to enable)
    mutation_types: List[str] = field(
        default_factory=lambda: [
            "identity_continuity",
            "locality",
            "invariant_subspace",
            "temporal_splice",
            "cycle_breaking",
        ]
    )

    # Validator committee
    num_validators: int = 8
    disagreement_threshold: float = 0.3
    gamma_thresholds: List[float] = field(
        default_factory=lambda: [0.1, 0.2, 0.15, 0.1, 0.15, 0.2, 0.1, 0.5]
    )

    # TAMG corruptor loss weights
    lambda_manifold: float = 1.0
    lambda_minimality: float = 0.3
    lambda_disagreement: float = 1.0
    lambda_diversity: float = 0.2
    lambda_artifact: float = 0.5

    # Manifold preservation
    manifold_epsilon: float = 0.1
    decode_threshold: float = 0.05  # max allowed reconstruction error

    # Operator training
    freeze_validators_during_corruptor_update: bool = True


@dataclass
class TrainingConfig:
    """General training configuration."""
    batch_size: int = 32
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    num_epochs: int = 100
    horizon: int = 25
    seed: int = 42
    log_every: int = 10
    save_every: int = 1000
    eval_every: int = 500
    checkpoint_dir: str = "outputs/checkpoints"
    gradient_clip_norm: float = 1.0
    num_workers: int = 4
    mixed_precision: bool = True


@dataclass
class ExperimentConfig:
    """Master experiment configuration."""
    env: EnvConfig = field(default_factory=EnvConfig)
    iwcm: IWCMConfig = field(default_factory=IWCMConfig)
    ac3: AC3Config = field(default_factory=AC3Config)
    tamg: TAMGConfig = field(default_factory=TAMGConfig)
    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    experiment_name: str = "default"
    description: str = ""
    use_wandb: bool = False
    use_tensorboard: bool = True


# ═══════════════════════════════════════════════════════════
# Config Loading Utilities
# ═══════════════════════════════════════════════════════════

def load_config(path: str) -> ExperimentConfig:
    """Load experiment config from YAML file, returning structured dataclass.

    Args:
        path: Path to YAML config file.

    Returns:
        ExperimentConfig dataclass with all nested configs.
    """
    with open(path, "r") as f:
        raw = yaml.safe_load(f)

    if HAS_OMEGACONF:
        cfg = OmegaConf.structured(ExperimentConfig)
        merged = OmegaConf.merge(cfg, OmegaConf.create(raw))
        return OmegaConf.to_object(merged)

    # Fallback: manual conversion without OmegaConf
    return _dict_to_config(raw)


def _dict_to_config(d: dict) -> ExperimentConfig:
    """Convert raw dict to ExperimentConfig dataclass (OmegaConf-free fallback)."""
    env = EnvConfig(**d.get("env", {}))
    iwcm = IWCMConfig(**d.get("iwcm", {}))
    ac3 = AC3Config(**d.get("ac3", {}))
    tamg = TAMGConfig(**d.get("tamg", {}))
    encoder = EncoderConfig(**d.get("encoder", {}))
    training = TrainingConfig(**d.get("training", {}))
    return ExperimentConfig(
        env=env, iwcm=iwcm, ac3=ac3, tamg=tamg,
        encoder=encoder, training=training,
        experiment_name=d.get("experiment_name", "default"),
        description=d.get("description", ""),
        use_wandb=d.get("use_wandb", False),
        use_tensorboard=d.get("use_tensorboard", True),
    )


def save_config(config: ExperimentConfig, path: str) -> None:
    """Save experiment config to YAML file."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    d = {
        "env": config.env.__dict__,
        "iwcm": config.iwcm.__dict__,
        "ac3": config.ac3.__dict__,
        "tamg": config.tamg.__dict__,
        "encoder": config.encoder.__dict__,
        "training": config.training.__dict__,
        "experiment_name": config.experiment_name,
        "description": config.description,
        "use_wandb": config.use_wandb,
        "use_tensorboard": config.use_tensorboard,
    }
    with open(path, "w") as f:
        yaml.dump(d, f, default_flow_style=False, sort_keys=False)
