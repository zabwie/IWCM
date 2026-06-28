#!/usr/bin/env python3
"""Generate cross-surface law generalization test data.

Creates train/test splits for each causal law with different surface forms.

Laws:
  1. Conservation: key duplicates (train) → box duplicates (test)
  2. Identity: object swaps under occlusion
  3. Causal locality: push affects distant objects
  4. Temporal order: cause before effect reversal
  5. Containment: object permanence under containers

Each law generates:
  - valid trajectories (no violation)
  - invalid trajectories with TRAIN surface form
  - invalid trajectories with TEST surface form (different appearance, same law)

GPU-optimized: generates data in parallel where possible.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
from typing import Dict, List, Tuple
from copy import deepcopy

from src.env.grid_world import GridWorld
from src.env.actions import Action
from src.env.data import encode_state, encode_action
from src.env.scenarios import Scenario, PREDEFINED_SCENARIOS
from src.ac3.mutations.grammar import SymbolicMutationGrammar, SymbolicTrajectory
from src.env.symbolic_state import SymbolicState
from src.ac3.oracle import SymbolicOracle


def generate_valid_trajectories(
    scenario_name: str, num: int, horizon: int, seed: int = 42,
) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Generate valid trajectories from a scenario."""
    scenario = Scenario.from_preset(scenario_name, 8)
    rng = np.random.RandomState(seed)
    trajectories = []

    for i in range(num):
        env = GridWorld(grid_size=8, objects_config=scenario.to_env_config(),
                       seed=int(rng.randint(0, 2**31)))
        env.reset()
        states, actions = [], []
        state = env.get_state()
        states.append(deepcopy(state))

        for t in range(horizon):
            valid_acts = env.get_valid_actions()
            if not valid_acts:
                break
            a = int(rng.choice(valid_acts))
            state, _, done, _ = env.step(a)
            states.append(deepcopy(state))
            actions.append(a)
            if done:
                break

        if len(states) >= horizon + 1 and len(actions) >= horizon:
            z0 = encode_state(states[0], 8)
            A = np.array([encode_action(a) for a in actions[:horizon]])
            Z = np.array([encode_state(s, 8) for s in states[1:horizon+1]])
            trajectories.append((z0, A, Z))

    return trajectories


def generate_conservation_violations(
    num: int, horizon: int, seed: int = 42,
) -> Tuple[List[Tuple], List[Tuple]]:
    """Generate conservation violations: train=key duplicates, test=box duplicates."""
    # Use multi_object scenario which has both keys and boxes
    scenario = Scenario.from_preset("multi_object", 8)
    rng = np.random.RandomState(seed)

    train_invalid = []  # key duplicates
    test_invalid = []   # box duplicates
    valid = generate_valid_trajectories("multi_object", num, horizon, seed)

    for i in range(num):
        env = GridWorld(grid_size=8, objects_config=scenario.to_env_config(),
                       seed=int(rng.randint(0, 2**31)))
        env.reset()
        states, actions = [], []
        state = env.get_state()
        states.append(deepcopy(state))

        for t in range(horizon):
            valid_acts = env.get_valid_actions()
            if not valid_acts:
                break
            a = int(rng.choice(valid_acts))
            state, _, done, _ = env.step(a)
            states.append(deepcopy(state))
            actions.append(a)
            if done:
                break

        if len(states) < horizon + 1:
            continue

        # Create key-duplicate violation (TRAIN surface)
        dup_states = deepcopy(states)
        keys = [oid for oid, tp in
                {k: guess_type(states[0], k) for k in states[0].get("objects", {})}.items()
                if tp == "key"]
        if keys:
            dup_key = rng.choice(keys)
            new_id = f"{dup_key}_dup"
            t_dup = rng.randint(horizon // 2, horizon)
            for t in range(t_dup, len(dup_states)):
                if dup_key in dup_states[t]["objects"]:
                    dup_states[t]["objects"][new_id] = deepcopy(dup_states[t]["objects"][dup_key])

            z0_d = encode_state(dup_states[0], 8)
            A_d = np.array([encode_action(a) for a in actions[:horizon]])
            Z_d = np.array([encode_state(s, 8) for s in dup_states[1:horizon+1]])
            train_invalid.append((z0_d, A_d, Z_d))

        # Create box-duplicate violation (TEST surface — different form, same law)
        box_states = deepcopy(states)
        boxes = [oid for oid in states[0].get("objects", {}).keys()
                if states[0]["objects"][oid].get("type") == "box"]
        if boxes:
            dup_box = rng.choice(boxes)
            new_id = f"{dup_box}_dup"
            t_dup = rng.randint(horizon // 2, horizon)
            for t in range(t_dup, len(box_states)):
                if dup_box in box_states[t]["objects"]:
                    box_states[t]["objects"][new_id] = deepcopy(box_states[t]["objects"][dup_box])

            z0_b = encode_state(box_states[0], 8)
            A_b = np.array([encode_action(a) for a in actions[:horizon]])
            Z_b = np.array([encode_state(s, 8) for s in box_states[1:horizon+1]])
            test_invalid.append((z0_b, A_b, Z_b))

    return train_invalid, test_invalid


def guess_type(state, obj_id):
    return state.get("objects", {}).get(obj_id, {}).get("type", "unknown")


def generate_identity_violations(
    num: int, horizon: int, seed: int = 42,
) -> Tuple[List[Tuple], List[Tuple]]:
    """Identity violations: object swaps under occlusion."""
    # Use multi_object which has keys, boxes for swaps
    scenario = Scenario.from_preset("multi_object", 8)
    rng = np.random.RandomState(seed)

    train_invalid, test_invalid = [], []
    valid = generate_valid_trajectories("multi_object", num, horizon, seed)

    for i in range(num):
        env = GridWorld(grid_size=8, objects_config=scenario.to_env_config(),
                       seed=int(rng.randint(0, 2**31)))
        env.reset()
        states, actions = [], []
        state = env.get_state()
        states.append(deepcopy(state))
        for t in range(horizon):
            valid_acts = env.get_valid_actions()
            if not valid_acts: break
            a = int(rng.choice(valid_acts))
            state, _, done, _ = env.step(a)
            states.append(deepcopy(state)); actions.append(a)
            if done: break
        if len(states) < horizon + 1: continue

        # Identity swap: red key <-> blue key (TRAIN) or agent <-> key (TEST)
        keys = [oid for oid in states[0].get("objects", {}).keys()
               if states[0]["objects"][oid].get("type") == "key"]
        if len(keys) >= 2:
            k1, k2 = keys[:2]
            # Train: swap key positions
            ts = deepcopy(states)
            t_swap = rng.randint(1, len(ts) - 2)
            for t in range(t_swap, len(ts)):
                if k1 in ts[t]["objects"] and k2 in ts[t]["objects"]:
                    ts[t]["objects"][k1]["pos"], ts[t]["objects"][k2]["pos"] = \
                        ts[t]["objects"][k2]["pos"], ts[t]["objects"][k1]["pos"]
            z0, A = encode_state(ts[0], 8), np.array([encode_action(a) for a in actions[:horizon]])
            Z = np.array([encode_state(s, 8) for s in ts[1:horizon+1]])
            train_invalid.append((z0, A, Z))

            # Test: swap key with box
            boxes = [oid for oid in states[0].get("objects", {}).keys()
                    if states[0]["objects"][oid].get("type") == "box"]
            if boxes:
                ts2 = deepcopy(states)
                kb = boxes[0]
                t_swap = rng.randint(1, len(ts2) - 2)
                for t in range(t_swap, len(ts2)):
                    if k1 in ts2[t]["objects"] and kb in ts2[t]["objects"]:
                        ts2[t]["objects"][k1]["pos"], ts2[t]["objects"][kb]["pos"] = \
                            ts2[t]["objects"][kb]["pos"], ts2[t]["objects"][k1]["pos"]
                z0_b = encode_state(ts2[0], 8)
                A_b = np.array([encode_action(a) for a in actions[:horizon]])
                Z_b = np.array([encode_state(s, 8) for s in ts2[1:horizon+1]])
                test_invalid.append((z0_b, A_b, Z_b))

    return train_invalid, test_invalid


def generate_locality_violations(
    num: int, horizon: int, seed: int = 42,
) -> Tuple[List[Tuple], List[Tuple]]:
    """Locality violations: push affects distant box (train), pickup affects door (test)."""
    scenario = Scenario.from_preset("multi_object", 8)
    rng = np.random.RandomState(seed)
    train_invalid, test_invalid = [], []
    valid = generate_valid_trajectories("multi_object", num, horizon, seed)

    for i in range(num):
        env = GridWorld(grid_size=8, objects_config=scenario.to_env_config(),
                       seed=int(rng.randint(0, 2**31)))
        env.reset()
        states, actions = [], []
        state = env.get_state()
        states.append(deepcopy(state))
        for t in range(horizon):
            valid_acts = env.get_valid_actions()
            if not valid_acts: break
            a = int(rng.choice(valid_acts))
            state, _, done, _ = env.step(a)
            states.append(deepcopy(state)); actions.append(a)
            if done: break
        if len(states) < horizon + 1: continue

        # Train: push causes distant box to move
        ts = deepcopy(states)
        t_loc = rng.randint(1, len(ts)-2)
        boxes = [oid for oid in ts[0].get("objects", {}).keys()
                if ts[0]["objects"][oid].get("type") == "box"]
        if boxes:
            b = rng.choice(boxes)
            for t in range(t_loc, len(ts)):
                if b in ts[t]["objects"]:
                    old = ts[t]["objects"][b].get("pos", (0,0))
                    nr = min(7, max(0, old[0] + rng.choice([-1, 1])))
                    nc = min(7, max(0, old[1] + rng.choice([-1, 1])))
                    ts[t]["objects"][b]["pos"] = (nr, nc)
            z0_l = encode_state(ts[0], 8)
            A_l = np.array([encode_action(a) for a in actions[:horizon]])
            Z_l = np.array([encode_state(s, 8) for s in ts[1:horizon+1]])
            train_invalid.append((z0_l, A_l, Z_l))

        # Test: pickup changes door state (different surface, same law)
        ts2 = deepcopy(states)
        doors = [oid for oid in ts2[0].get("objects", {}).keys()
                if ts2[0]["objects"][oid].get("type") == "door"]
        if doors:
            d = rng.choice(doors)
            for t in range(1, len(ts2)):
                if d in ts2[t].get("door_states", {}):
                    ts2[t]["door_states"][d] = not ts2[t-1].get("door_states", {}).get(d, False)
            z0_d = encode_state(ts2[0], 8)
            A_d = np.array([encode_action(a) for a in actions[:horizon]])
            Z_d = np.array([encode_state(s, 8) for s in ts2[1:horizon+1]])
            test_invalid.append((z0_d, A_d, Z_d))

    return train_invalid, test_invalid


def generate_temporal_violations(
    num: int, horizon: int, seed: int = 42,
) -> Tuple[List[Tuple], List[Tuple]]:
    """Temporal order: door opens before key pickup (train), other reversals (test)."""
    rng = np.random.RandomState(seed)
    train_invalid, test_invalid = [], []

    for i in range(num):
        scenario = Scenario.from_preset("key_door_simple", 8)
        env = GridWorld(grid_size=8, objects_config=scenario.to_env_config(),
                       seed=int(rng.randint(0, 2**31)))
        env.reset()
        states, actions = [], []
        state = env.get_state()
        states.append(deepcopy(state))
        for t in range(horizon):
            valid_acts = env.get_valid_actions()
            if not valid_acts: break
            a = int(rng.choice(valid_acts))
            state, _, done, _ = env.step(a)
            states.append(deepcopy(state)); actions.append(a)
            if done: break
        if len(states) < horizon + 1: continue

        # Train: door opens before key pickup
        ts = deepcopy(states)
        doors = [oid for oid in ts[0].get("objects", {}).keys()
                if ts[0]["objects"][oid].get("type") == "door"]
        if doors and len(ts) > 2:
            d = doors[0]
            for t in range(len(ts)):
                ts[t]["door_states"] = ts[t].get("door_states", {})
                ts[t]["door_states"][d] = True  # door open throughout (invalid)
            z0 = encode_state(ts[0], 8)
            A = np.array([encode_action(a) for a in actions[:horizon]])
            Z = np.array([encode_state(s, 8) for s in ts[1:horizon+1]])
            train_invalid.append((z0, A, Z))

        # Test: swap two consecutive states (different form, same law)
        ts2 = deepcopy(states)
        if len(ts2) >= 4:
            t_mid = len(ts2) // 2
            ts2[t_mid], ts2[t_mid+1] = ts2[t_mid+1], ts2[t_mid]
            z0_t = encode_state(ts2[0], 8)
            actions2 = list(actions)
            if t_mid < len(actions2)-1:
                actions2[t_mid], actions2[t_mid+1] = actions2[t_mid+1], actions2[t_mid]
            A_t = np.array([encode_action(a) for a in actions2[:horizon]])
            Z_t = np.array([encode_state(s, 8) for s in ts2[1:horizon+1]])
            test_invalid.append((z0_t, A_t, Z_t))

    return train_invalid, test_invalid


def main():
    import argparse, pickle
    parser = argparse.ArgumentParser()
    parser.add_argument("--num", type=int, default=500, help="Trajectories per type")
    parser.add_argument("--horizon", type=int, default=25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default="data/cross_surface.pkl")
    args = parser.parse_args()

    print("Generating cross-surface law generalization test data...")
    print(f"  Num per type: {args.num}, Horizon: {args.horizon}")

    valid_trajs = generate_valid_trajectories("key_door_simple", args.num, args.horizon, args.seed)

    # Conservation
    print("  Conservation violations...")
    cons_train, cons_test = generate_conservation_violations(args.num, args.horizon, args.seed+1)

    # Identity
    print("  Identity violations...")
    id_train, id_test = generate_identity_violations(args.num, args.horizon, args.seed+2)

    # Locality
    print("  Locality violations...")
    loc_train, loc_test = generate_locality_violations(args.num, args.horizon, args.seed+3)

    # Temporal
    print("  Temporal violations...")
    temp_train, temp_test = generate_temporal_violations(args.num, args.horizon, args.seed+4)

    data = {
        "valid": valid_trajs,
        "train": {"conservation": cons_train, "identity": id_train,
                  "locality": loc_train, "temporal": temp_train},
        "test": {"conservation": cons_test, "identity": id_test,
                "locality": loc_test, "temporal": temp_test},
    }

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "wb") as f:
        pickle.dump(data, f)

    print(f"\nSaved to {args.output}")
    print(f"  Valid: {len(valid_trajs)}")
    for law in ["conservation", "identity", "locality", "temporal"]:
        print(f"  {law}: train={len(data['train'][law])}, test={len(data['test'][law])}")
    print("Done.")


if __name__ == "__main__":
    main()
