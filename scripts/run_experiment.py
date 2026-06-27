#!/usr/bin/env python3
"""Unified experiment runner for IWCM experiments.

Usage:
    python scripts/run_experiment.py --exp exp1 --model c --horizon 25 --epochs 100
    python scripts/run_experiment.py --exp exp2 --model d --horizon 25 --epochs 150
"""

import argparse
import sys
from pathlib import Path
import torch
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.config import load_config, ExperimentConfig
from src.utils.seed import set_seed
from src.utils.logging import MetricsLogger
from src.env.data import generate_dataset, CounterfactualDataset
from src.env.scenarios import Scenario, PREDEFINED_SCENARIOS
from src.iwcm.model import IWCM
from src.iwcm.energy import IWCMEnergy
from src.ac3.trainer import AC3Trainer
from src.ac3.corruptor import AC3Corruptor
from src.ac3.oracle import SymbolicOracle
from src.tamg.trainer import TAMGTrainer
from src.tamg.corruptor import TAMGCorruptor
from src.tamg.validators.committee import ValidatorCommittee
from src.encoder.video_encoder import VideoEncoder
from src.encoder.decoder import VideoDecoder
from src.metrics.evaluation import evaluate_model


def run_experiment_1(config: ExperimentConfig, logger: MetricsLogger):
    """Run Experiment 1: Symbolic grid world (IWCM + AC3)."""
    print("=" * 60)
    print(f"Experiment 1: {config.experiment_name}")
    print(f"Description: {config.description}")
    print("=" * 60)

    set_seed(config.training.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # State dimension from environment
    grid_size = config.env.grid_size
    d_state = grid_size * grid_size * 4  # 4 channels of state encoding
    d_action = 11

    # Generate data
    scenario_names = list(PREDEFINED_SCENARIOS.keys())[:5]
    dataset = generate_dataset(
        scenario_names=scenario_names,
        horizon=config.training.horizon,
        num_trajectories=10000,
        grid_size=grid_size,
        seed=config.training.seed,
    )
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=config.training.batch_size,
        shuffle=True, num_workers=config.training.num_workers,
    )

    # Initialize world model
    world_model = IWCM(
        d_state=d_state,
        d_action=d_action,
        hidden_dim=config.iwcm.hidden_dim,
        lambdas={
            "boundary": config.iwcm.lambda_boundary,
            "local": config.iwcm.lambda_local,
            "invariant": config.iwcm.lambda_invariant,
            "effect": config.iwcm.lambda_effect,
            "counterfactual": config.iwcm.lambda_counterfactual,
        },
        solver_steps=config.iwcm.solver_steps,
        solver_lr=config.iwcm.solver_lr,
        use_refinement=config.iwcm.use_learned_refinement,
    )

    # Initialize corruptor and oracle
    corruptor = AC3Corruptor(d_state=d_state, hidden_dim=128)
    oracle = SymbolicOracle()

    # Trainer
    trainer = AC3Trainer(
        world_model=world_model,
        corruptor=corruptor,
        oracle=oracle,
        logger=logger,
        device=device,
        lr_world=config.training.learning_rate,
        lr_corruptor=config.training.learning_rate,
        num_mutations_per_sample=config.ac3.num_mutations_per_sample,
        top_k_hard=config.ac3.top_k_hard,
        accept_low=config.ac3.accept_low,
        accept_high=config.ac3.accept_high,
        lambda_valid=config.ac3.lambda_valid_energy,
        lambda_invalid=config.ac3.lambda_invalid_energy,
        lambda_repair=config.ac3.lambda_repair,
        lambda_accept=config.ac3.lambda_accept,
        lambda_minimal=config.ac3.lambda_minimal,
        lambda_diversity=config.ac3.lambda_diversity,
    )

    # Training loop
    for epoch in range(config.training.num_epochs):
        metrics = trainer.train_epoch(dataloader)
        print(f"Epoch {epoch + 1}/{config.training.num_epochs}: "
              f"loss_w={metrics['loss_world']:.4f}, "
              f"loss_c={metrics['loss_corruptor']:.4f}, "
              f"accept={metrics['accept_mean']:.3f}, "
              f"violation={metrics['violation_rate']:.2f}")

        if (epoch + 1) % config.training.save_every == 0:
            ckpt_path = f"{config.training.checkpoint_dir}/exp1_{config.experiment_name}_epoch{epoch+1}.pt"
            world_model.save(ckpt_path)

    # Evaluation
    print("\nRunning evaluation...")
    eval_data = {"valid_trajs": list(dataset)}
    results = evaluate_model(world_model, eval_data)
    for metric, value in results.items():
        print(f"  {metric}: {value}")

    print("\nExperiment 1 complete.")
    return world_model, results


def run_experiment_2(config: ExperimentConfig, logger: MetricsLogger):
    """Run Experiment 2: Video environment (TAMG + Validator Committee)."""
    print("=" * 60)
    print(f"Experiment 2: {config.experiment_name}")
    print(f"Description: {config.description}")
    print("=" * 60)

    set_seed(config.training.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    grid_size = config.env.grid_size
    d_state = grid_size * grid_size * 4
    d_action = 11

    # Generate video data
    scenario_names = list(PREDEFINED_SCENARIOS.keys())[:5]
    dataset = generate_dataset(
        scenario_names=scenario_names,
        horizon=config.training.horizon,
        num_trajectories=5000,
        grid_size=grid_size,
        seed=config.training.seed,
    )
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=config.training.batch_size,
        shuffle=True, num_workers=config.training.num_workers,
    )

    # Encoder/Decoder
    encoder = VideoEncoder(
        frame_size=config.encoder.frame_size,
        in_channels=3,
        cnn_channels=tuple(config.encoder.cnn_channels),
        num_slots=config.encoder.num_slots,
        slot_dim=config.encoder.slot_dim,
        slot_iters=config.encoder.slot_iters,
    )
    decoder = VideoDecoder(
        slot_dim=config.encoder.slot_dim,
        frame_size=config.encoder.frame_size,
    )

    # World model (operates on slot latents)
    world_model = IWCM(
        d_state=config.encoder.slot_dim,
        d_action=d_action,
        hidden_dim=config.iwcm.hidden_dim,
        solver_steps=config.iwcm.solver_steps,
        solver_lr=config.iwcm.solver_lr,
        use_refinement=config.iwcm.use_learned_refinement,
    )

    # TAMG corruptor
    corruptor = TAMGCorruptor(
        slot_dim=config.encoder.slot_dim,
        content_dim=config.encoder.content_dim,
        num_operators=config.tamg.num_operators,
        d_action=d_action,
    )

    # Validator committee
    validators = ValidatorCommittee(
        d_state=config.encoder.slot_dim,
        d_action=d_action,
        content_dim=config.encoder.content_dim,
    )

    # Trainer
    trainer = TAMGTrainer(
        world_model=world_model,
        corruptor=corruptor,
        validators=validators,
        encoder=encoder,
        decoder=decoder,
        logger=logger,
        device=device,
        lambda_valid=config.ac3.lambda_valid_energy,
        lambda_invalid=config.ac3.lambda_invalid_energy,
        lambda_repair=config.ac3.lambda_repair,
        lambda_disagreement=config.tamg.lambda_disagreement,
        lambda_manifold=config.tamg.lambda_manifold,
        lambda_minimality=config.tamg.lambda_minimality,
        lambda_diversity=config.tamg.lambda_diversity,
        lambda_artifact=config.tamg.lambda_artifact,
        disagreement_threshold=config.tamg.disagreement_threshold,
    )

    # Training
    for epoch in range(config.training.num_epochs):
        metrics = trainer.train_epoch(dataloader)
        print(f"Epoch {epoch + 1}/{config.training.num_epochs}: "
              f"loss_w={metrics['loss_world']:.4f}, "
              f"loss_c={metrics['loss_corruptor']:.4f}, "
              f"accept={metrics['accept_mean']:.3f}, "
              f"D={metrics['d_score_mean']:.3f}")

        if (epoch + 1) % config.training.save_every == 0:
            world_model.save(
                f"{config.training.checkpoint_dir}/exp2_{config.experiment_name}_epoch{epoch+1}.pt",
            )

    print("\nExperiment 2 complete.")
    return world_model, {}


def main():
    parser = argparse.ArgumentParser(description="Run IWCM experiments")
    parser.add_argument("--exp", type=str, required=True,
                       choices=["exp1", "exp2"],
                       help="Experiment to run")
    parser.add_argument("--model", type=str, required=True,
                       choices=["a", "b", "c", "d"],
                       help="Model variant")
    parser.add_argument("--horizon", type=int, default=25)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto",
                       choices=["auto", "cuda", "cpu"])
    parser.add_argument("--wandb", action="store_true")

    args = parser.parse_args()

    # Load config
    config_path = f"configs/{args.exp}/model_{args.model}.yaml"
    config = load_config(config_path)

    # Override from CLI
    config.training.horizon = args.horizon
    if args.epochs:
        config.training.num_epochs = args.epochs
    if args.batch_size:
        config.training.batch_size = args.batch_size
    config.training.seed = args.seed
    config.use_wandb = args.wandb

    # Logger
    logger = MetricsLogger(
        log_dir=f"outputs/logs/{args.exp}_model_{args.model}",
        project_name="iwcm",
        run_name=f"{args.exp}_model_{args.model}",
        use_wandb=args.wandb,
        use_tensorboard=True,
    )

    # Run
    if args.exp == "exp1":
        run_experiment_1(config, logger)
    else:
        run_experiment_2(config, logger)

    logger.close()


if __name__ == "__main__":
    main()
