#!/usr/bin/env python3
"""Data generation CLI for IWCM experiments.

Usage:
    python scripts/generate_data.py --scenario key_door_simple --horizon 25 --num 1000 --seed 42
    python scripts/generate_data.py --all --horizon 25 --num 5000 --seed 42
    python scripts/generate_data.py --counterfactuals --horizon 25 --num 500 --seed 42
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.env.scenarios import (
    Scenario, PREDEFINED_SCENARIOS,
)
from src.env.data import (
    TrajectoryDataset, CounterfactualDataset,
    generate_dataset, save_dataset,
)


def main():
    parser = argparse.ArgumentParser(
        description="Generate training data for IWCM experiments"
    )
    parser.add_argument(
        "--scenario", type=str, default=None,
        help=f"Scenario name: {list(PREDEFINED_SCENARIOS.keys())}",
    )
    parser.add_argument(
        "--all", action="store_true",
        help="Generate data for all predefined scenarios",
    )
    parser.add_argument(
        "--counterfactuals", action="store_true",
        help="Generate counterfactual pairs instead of regular trajectories",
    )
    parser.add_argument(
        "--horizon", type=int, default=25,
        help="Trajectory horizon (default: 25)",
    )
    parser.add_argument(
        "--num", type=int, default=10000,
        help="Number of trajectories to generate (default: 10000)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--output", type=str, default="data/",
        help="Output directory (default: data/)",
    )
    parser.add_argument(
        "--policy", type=str, default="mixed",
        choices=["random", "expert", "mixed"],
        help="Action selection policy (default: mixed)",
    )
    parser.add_argument(
        "--grid-size", type=int, default=8,
        help="Grid size (default: 8)",
    )

    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    scenario_names = (
        list(PREDEFINED_SCENARIOS.keys()) if args.all
        else [args.scenario] if args.scenario
        else ["key_door_simple"]
    )

    print(f"Generating data:")
    print(f"  Scenarios: {scenario_names}")
    print(f"  Horizon: {args.horizon}")
    print(f"  Trajectories: {args.num}")
    print(f"  Policy: {args.policy}")
    print(f"  Seed: {args.seed}")
    print(f"  Output: {output_dir.resolve()}")

    if args.counterfactuals:
        # Generate counterfactual pairs
        from src.env.scenarios import generate_counterfactual_pairs
        pairs = generate_counterfactual_pairs(
            Scenario.from_preset(scenario_names[0], args.grid_size),
            horizon=args.horizon,
            num_pairs=args.num,
            seed=args.seed,
        )
        dataset = CounterfactualDataset(
            pairs, args.horizon, args.grid_size,
        )
        filename = f"counterfactual_{scenario_names[0]}_h{args.horizon}_n{args.num}.pkl"
    else:
        dataset = generate_dataset(
            scenario_names=scenario_names,
            horizon=args.horizon,
            num_trajectories=args.num,
            policy=args.policy,
            grid_size=args.grid_size,
            seed=args.seed,
        )
        filename = f"trajectories_{'all' if args.all else scenario_names[0]}_h{args.horizon}_n{args.num}.pkl"

    output_path = output_dir / filename
    save_dataset(dataset, str(output_path))
    print(f"\nSaved {len(dataset)} items to {output_path}")
    print("Done.")


if __name__ == "__main__":
    main()
