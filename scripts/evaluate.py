#!/usr/bin/env python3
"""Evaluate a trained IWCM model.

Usage:
    python scripts/evaluate.py --checkpoint outputs/checkpoints/model.pt --exp exp1
"""

import argparse
import sys
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.iwcm.model import IWCM
from src.utils.config import load_config
from src.utils.seed import set_seed
from src.env.data import generate_dataset
from src.env.scenarios import PREDEFINED_SCENARIOS
from src.metrics.evaluation import evaluate_model


def main():
    parser = argparse.ArgumentParser(description="Evaluate IWCM model")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--exp", type=str, default="exp1",
                       choices=["exp1", "exp2"])
    parser.add_argument("--horizon", type=int, default=25)
    parser.add_argument("--grid-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    d_state = args.grid_size * args.grid_size * 4
    d_action = 11

    # Load model
    model = IWCM(d_state=d_state, d_action=d_action, hidden_dim=256)
    model.load(args.checkpoint, map_location=device)
    model.to(device)
    model.eval()

    print(f"Model loaded from {args.checkpoint}")
    print(f"Parameters: {model.count_parameters_str()}")

    # Generate evaluation data
    scenario_names = list(PREDEFINED_SCENARIOS.keys())[:5]
    dataset = generate_dataset(
        scenario_names=scenario_names,
        horizon=args.horizon,
        num_trajectories=500,
        grid_size=args.grid_size,
        seed=args.seed + 1000,
    )

    eval_data = {"valid_trajs": list(dataset)}
    results = evaluate_model(model, eval_data)

    print("\nEvaluation Results:")
    print("-" * 50)
    for metric, value in results.items():
        if isinstance(value, dict):
            print(f"{metric}:")
            for k, v in value.items():
                print(f"  {k}: {v}")
        else:
            print(f"{metric}: {value}")


if __name__ == "__main__":
    main()
