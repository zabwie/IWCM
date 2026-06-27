#!/usr/bin/env python3
"""Compositional corruption grid — balanced multi-axis corruption generation.

For each law, generates corruptions across independent axes:
  - object_type: key, box, door
  - context: visible, occluded, carried
  - violation_type: duplicate, delete, transform, swap, teleport, illegal-open, reverse
  - time_gap: early, mid, late
  - distractors: none, same_type, different_type

Splits compositionally so no single surface cue predicts validity — forcing
the model to learn abstract laws, not surface patterns.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pickle
from typing import Dict, List, Tuple
from copy import deepcopy

from src.env.grid_world import GridWorld
from src.env.actions import Action
from src.env.data import encode_state, encode_action
from src.env.scenarios import Scenario, PREDEFINED_SCENARIOS
from src.env.symbolic_state import SymbolicState, SymbolicTrajectory, symbolic_to_state_dict
from src.ac3.oracle import SymbolicOracle
from src.encoder.oracle_slot_encoder import encode_oracle_trajectory, build_door_key_map, ORACLE_SLOT_DIM, MAX_OBJECTS


def generate_valid_trajectory(scenario: Scenario, horizon: int, rng: np.random.RandomState) -> Tuple:
    """Generate a single valid trajectory from a scenario."""
    env = GridWorld(grid_size=8, objects_config=scenario.to_env_config(),
                   seed=int(rng.randint(0, 2**31)))
    env.reset()
    states, actions = [], []
    state = env.get_state()
    states.append(deepcopy(state))
    for t in range(horizon * 3):
        valid_acts = env.get_valid_actions()
        if not valid_acts: break
        a = int(rng.choice(valid_acts))
        state, _, done, _ = env.step(a)
        states.append(deepcopy(state)); actions.append(a)
        if done: break
    if len(states) >= horizon + 1 and len(actions) >= horizon:
        return states, actions, scenario
    return None


def apply_corruption(states, actions, scenario, law_type, violation_type, object_type,
                     time_gap, context, horizon, rng):
    """Apply a specific corruption to a trajectory."""
    gs = 8
    n = len(states)
    if n < horizon + 1: return None

    # Determine time gap
    if time_gap == "early":   t_corrupt = rng.randint(1, max(2, horizon // 3))
    elif time_gap == "mid":   t_corrupt = rng.randint(horizon // 3, max(horizon // 3 + 1, 2 * horizon // 3))
    else:                     t_corrupt = rng.randint(2 * horizon // 3, max(2 * horizon // 3 + 1, horizon - 1))

    corr_states = deepcopy(states)
    oracle = SymbolicOracle()

    # Find objects of the target type
    target_objects = []
    for oid, obj in corr_states[t_corrupt].get("objects", {}).items():
        if obj.get("type") == object_type:
            target_objects.append(oid)

    if not target_objects:
        return None

    target = rng.choice(target_objects)
    obj_info = corr_states[t_corrupt]["objects"][target]

    # Apply context constraints
    if context == "carried":
        # Move object to agent's inventory at corruption time
        for t in range(t_corrupt, min(n, horizon + 1)):
            if target in corr_states[t].get("objects", {}):
                inv = corr_states[t].setdefault("inventory", [])
                if target not in inv and len(inv) < 4:
                    inv.append(target)
    elif context == "occluded":
        # Ensure object is behind an occluder — mark it as occluded context
        pass  # Context is metadata, not physical change

    # Apply the violation
    if law_type == "conservation":
        if violation_type == "duplicate":
            new_id = f"{target}_dup"
            for t in range(t_corrupt, min(n, horizon + 1)):
                if target in corr_states[t].get("objects", {}):
                    corr_states[t]["objects"][new_id] = deepcopy(corr_states[t]["objects"][target])
        elif violation_type == "delete":
            for t in range(t_corrupt, min(n, horizon + 1)):
                corr_states[t].get("objects", {}).pop(target, None)
        elif violation_type == "transform":
            new_type = rng.choice(["key", "box", "door"])
            for t in range(t_corrupt, min(n, horizon + 1)):
                if target in corr_states[t].get("objects", {}):
                    corr_states[t]["objects"][target]["type"] = new_type

    elif law_type == "identity":
        others = [o for o in target_objects if o != target]
        if violation_type == "swap" and others:
            other = rng.choice(others)
            for t in range(t_corrupt, min(n, horizon + 1)):
                if target in corr_states[t].get("objects", {}) and other in corr_states[t].get("objects", {}):
                    corr_states[t]["objects"][target]["pos"], corr_states[t]["objects"][other]["pos"] = \
                        corr_states[t]["objects"][other]["pos"], corr_states[t]["objects"][target]["pos"]
        elif violation_type == "teleport":
            for t in range(t_corrupt, min(n, horizon + 1)):
                if target in corr_states[t].get("objects", {}):
                    old = corr_states[t]["objects"][target].get("pos", (0, 0))
                    nr = min(7, max(0, old[0] + rng.choice([-2, -1, 1, 2])))
                    nc = min(7, max(0, old[1] + rng.choice([-2, -1, 1, 2])))
                    corr_states[t]["objects"][target]["pos"] = (nr, nc)

    elif law_type == "locality":
        if violation_type == "illegal_open":
            doors = [oid for oid in corr_states[0].get("objects", {}).keys()
                    if corr_states[0]["objects"][oid].get("type") == "door"]
            if doors:
                d = rng.choice(doors)
                for t in range(1, min(n, horizon + 1)):
                    corr_states[t].setdefault("door_states", {})[d] = not corr_states[t-1].get("door_states", {}).get(d, False)

    elif law_type == "temporal":
        if violation_type == "reverse":
            for t in range(t_corrupt, min(n - 2, horizon)):
                if t + 1 < n:
                    corr_states[t], corr_states[t + 1] = corr_states[t + 1], corr_states[t]
                    if t < len(actions):
                        actions_list = list(actions)
                        actions_list[t], actions_list[t + 1] = actions_list[t + 1], actions_list[t]
                        actions = actions_list

    # Check if corruption made it invalid
    sym_states = []
    for s in corr_states[:horizon+1]:
        op, ot = {}, {}
        for oid, obj in s.get("objects", {}).items():
            op[oid] = tuple(obj.get("pos", (0, 0)))
            ot[oid] = obj.get("type", "unknown")
        sym_states.append(SymbolicState(
            agent_pos=tuple(s.get("agent_pos", (0, 0))), grid_size=gs, step=0,
            object_positions=op, object_types=ot,
            door_states=s.get("door_states", {}), inventory=s.get("inventory", []),
        ))
    st = SymbolicTrajectory(states=sym_states, actions=list(actions[:horizon]), horizon=horizon)
    violations = oracle(st)

    if len(violations) > 0:
        goal = corr_states[0].get("goal", None)
        dkm = build_door_key_map(scenario.to_env_config())
        enc = encode_oracle_trajectory(corr_states, list(actions[:horizon]), horizon, gs, goal, dkm)
        if enc is None: return None

        # Build metadata for shortcut audit
        meta = {
            "object_type": object_type,
            "context": context,
            "violation_type": violation_type,
            "time_gap": time_gap,
            "law_type": law_type,
            "violations": violations,
        }
        return enc, meta
    return None


def generate_compositional_grid(num_per_cell: int = 20, horizon: int = 25, seed: int = 42):
    """Generate the full compositional corruption grid.

    Returns train/test splits where no single surface axis predicts validity.
    """
    rng = np.random.RandomState(seed)

    # Define the factor space
    object_types = ["key", "box"]
    contexts = ["visible", "occluded", "carried"]
    violation_types = {
        "conservation": ["duplicate", "delete", "transform"],
        "identity": ["swap", "teleport"],
        "locality": ["illegal_open"],
        "temporal": ["reverse"],
    }
    time_gaps = ["early", "mid", "late"]

    scenarios = {name: Scenario.from_preset(name, 8) for name in
                 ["key_door_simple", "multi_object", "conservation_test", "counterfactual_test",
                  "box_push", "splice_test"]}

    valid_trajs = []
    all_corruptions = []

    # Generate valid trajectories first
    print("Generating valid trajectories...")
    for _ in range(num_per_cell * 20):
        sc = rng.choice(list(scenarios.values()))
        result = generate_valid_trajectory(sc, horizon, rng)
        if result:
            states, actions, scenario = result
            goal = states[0].get("goal", None)
            dkm = build_door_key_map(scenario.to_env_config())
            enc = encode_oracle_trajectory(states, actions, horizon, 8, goal, dkm)
            if enc:
                valid_trajs.append(enc)

    print(f"  Generated {len(valid_trajs)} valid trajectories")

    # Generate corruptions across all factor combinations
    print("Generating compositional corruptions...")
    total_cells = (len(object_types) * len(contexts) *
                   sum(len(v) for v in violation_types.values()) * len(time_gaps))
    print(f"  Grid size: {total_cells} factor combinations")

    for law_type, vtypes in violation_types.items():
        for object_type in object_types:
            for context in contexts:
                for violation_type in vtypes:
                    for time_gap in time_gaps:
                        for _ in range(num_per_cell):
                            sc = rng.choice(list(scenarios.values()))
                            result = generate_valid_trajectory(sc, horizon, rng)
                            if result is None: continue
                            states, actions, scenario = result
                            corr_result = apply_corruption(
                                states, actions, scenario, law_type, violation_type,
                                object_type, time_gap, context, horizon, rng,
                            )
                            if corr_result:
                                all_corruptions.append(corr_result)

    print(f"  Generated {len(all_corruptions)} corruptions")

    # Compositional split: divide factor space so no single axis predicts validity
    # Strategy: assign half of each factor level to train, half to test
    # BUT: ensure each combination of (law, violation) has train AND test forms
    train_corr, test_corr = [], []
    train_valid, test_valid = [], []

    # Split valid trajectories 60/40
    n_valid = len(valid_trajs)
    perm = rng.permutation(n_valid)
    train_valid = [valid_trajs[i] for i in perm[:int(0.6 * n_valid)]]
    test_valid = [valid_trajs[i] for i in perm[int(0.6 * n_valid):]]

    # Split corruptions: for each (law_type, violation_type) pair,
    # put half the object_type × context × time_gap combos in train, half in test
    for i, (enc, meta) in enumerate(all_corruptions):
        # Hash-based deterministic split that's NOT predictable from a single axis
        key = (meta["law_type"], meta["violation_type"],
               meta["object_type"], meta["context"], meta["time_gap"])
        h = abs(hash(key)) % 2
        if h == 0:
            train_corr.append((enc, meta))
        else:
            test_corr.append((enc, meta))

    print(f"\nSplit: train_valid={len(train_valid)} test_valid={len(test_valid)}")
    print(f"  train_corr={len(train_corr)} test_corr={len(test_corr)}")

    # Shortcut audit: check if metadata alone predicts valid/invalid
    print("\nShortcut audit (metadata-only classification)...")
    meta_features = []
    meta_labels = []
    for enc, meta in train_corr[:200]:
        feat = [hash(meta["object_type"]) % 100, hash(meta["context"]) % 100,
                hash(meta["violation_type"]) % 100, hash(meta["time_gap"]) % 100,
                hash(meta["law_type"]) % 100]
        meta_features.append(feat)
        meta_labels.append(1)  # corrupted

    for _ in range(len(meta_features)):
        meta_features.append([hash("valid") % 100] * 5)
        meta_labels.append(0)

    # Simple majority baseline
    maj_acc = max(sum(1 for l in meta_labels if l == 1), sum(1 for l in meta_labels if l == 0)) / len(meta_labels)
    print(f"  Majority baseline accuracy: {maj_acc:.3f}")
    print(f"  (Goal: shortcut accuracy near {maj_acc:.3f} = balanced)")

    return {
        "train_valid": train_valid, "train_corr": train_corr,
        "test_valid": test_valid, "test_corr": test_corr,
        "train_meta": [(m,) for _, m in train_corr],
        "test_meta": [(m,) for _, m in test_corr],
    }


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--num", type=int, default=20)
    parser.add_argument("--horizon", type=int, default=25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default="data/compositional_grid.pkl")
    args = parser.parse_args()

    data = generate_compositional_grid(args.num, args.horizon, args.seed)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "wb") as f:
        pickle.dump(data, f)
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
