#!/usr/bin/env python3
"""Generate explicit cross-surface corruption training pairs.

For each law type (conservation, identity, locality, temporal),
generates BOTH train-surface and test-surface corruptions,
paired with their corresponding valid trajectories.

This ensures the model sees ALL corruption forms during training,
which is necessary for cross-surface law generalization.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
import pickle
from typing import Dict, List, Tuple
from copy import deepcopy

from src.env.grid_world import GridWorld
from src.env.actions import Action
from src.env.data import encode_state, encode_action
from src.env.scenarios import Scenario, PREDEFINED_SCENARIOS
from src.env.symbolic_state import SymbolicState, SymbolicTrajectory, symbolic_to_state_dict
from src.ac3.oracle import SymbolicOracle
from src.ac3.mutations.grammar import SymbolicMutationGrammar


def generate_valid(scenario: Scenario, horizon: int, num: int, seed: int) -> List[Tuple]:
    """Generate valid trajectories from a scenario."""
    rng = np.random.RandomState(seed)
    trajs = []
    for i in range(num):
        env = GridWorld(grid_size=8, objects_config=scenario.to_env_config(), seed=int(rng.randint(0, 2**31)))
        env.reset()
        states, actions = [], []
        state = env.get_state()
        states.append(deepcopy(state))
        for t in range(horizon * 3):
            valid = env.get_valid_actions()
            if not valid: break
            a = int(rng.choice(valid))
            state, _, done, _ = env.step(a)
            states.append(deepcopy(state)); actions.append(a)
            if done: break
        if len(states) >= horizon + 1 and len(actions) >= horizon:
            z0 = encode_state(states[0], 8)
            A = np.array([encode_action(a) for a in actions[:horizon]])
            Z = np.array([encode_state(s, 8) for s in states[1:horizon+1]])
            trajs.append((z0, A, Z))
    return trajs


def apply_mutation_to_trajectory(states, actions, mutation_type, rng, horizon=25):
    """Apply a specific mutation type to a trajectory, returning encoded (z0, A, Z)."""
    gs = 8

    # Build SymbolicTrajectory from states+actions
    sym_states = []
    for s in states[:horizon+1]:
        obj_pos = {}
        obj_types = {}
        for oid, obj in s.get("objects", {}).items():
            obj_pos[oid] = tuple(obj["pos"])
            obj_types[oid] = obj.get("type", "unknown")
        sym_states.append(SymbolicState(
            agent_pos=tuple(s["agent_pos"]), grid_size=gs, step=0,
            object_positions=obj_pos, object_types=obj_types,
            door_states=dict(s.get("door_states", {})),
            inventory=list(s.get("inventory", [])),
        ))
    st = SymbolicTrajectory(states=sym_states, actions=list(actions[:horizon]), horizon=horizon)

    # Find the mutation
    grammar = SymbolicMutationGrammar()
    mutation_names = [m.name for m in grammar.mutations]

    for mutation in grammar.mutations:
        if mutation.name == mutation_type:
            corrupted_traj = mutation.apply(st, rng)
            # Convert back to state dicts and encode
            corrupted_dicts = [symbolic_to_state_dict(ss) for ss in corrupted_traj.states]
            if len(corrupted_dicts) < horizon + 1:
                return None
            z0_c = encode_state(corrupted_dicts[0], gs)
            A_c = np.array([encode_action(a) for a in corrupted_traj.actions[:horizon]])
            Z_c = np.array([encode_state(s, gs) for s in corrupted_dicts[1:horizon+1]])
            return (z0_c, A_c, Z_c)
    return None


def generate_conservation_pairs(num, horizon, seed):
    """Train: key duplication. Test: box duplication."""
    scenario = Scenario.from_preset("multi_object", 8)
    oracle = SymbolicOracle()
    rng = np.random.RandomState(seed)

    train_pairs, test_pairs = [], []

    for i in range(num):
        env = GridWorld(grid_size=8, objects_config=scenario.to_env_config(),
                       seed=int(rng.randint(0, 2**31)))
        env.reset()
        states, actions = [], []
        state = env.get_state()
        states.append(deepcopy(state))
        for t in range(horizon * 3):
            valid = env.get_valid_actions()
            if not valid: break
            a = int(rng.choice(valid))
            state, _, done, _ = env.step(a)
            states.append(deepcopy(state)); actions.append(a)
            if done: break
        if len(states) < horizon + 1: continue

        # Train: duplicate a key
        key_states = deepcopy(states)
        keys = [oid for oid in states[0].get("objects", {}).keys()
               if states[0]["objects"][oid].get("type") == "key"]
        if keys:
            dup_key = rng.choice(keys)
            new_id = f"{dup_key}_dup"
            t_dup = rng.randint(horizon // 2, horizon)
            for t in range(t_dup, len(key_states)):
                if dup_key in key_states[t]["objects"]:
                    key_states[t]["objects"][new_id] = deepcopy(key_states[t]["objects"][dup_key])
            z0, A = encode_state(key_states[0], 8), np.array([encode_action(a) for a in actions[:horizon]])
            Z = np.array([encode_state(s, 8) for s in key_states[1:horizon+1]])
            if len(oracle(SymbolicTrajectory.from_env_trajectory(key_states, actions))) > 0:
                train_pairs.append((z0, A, Z))

        # Test: duplicate a box
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
            z0_b, A_b = encode_state(box_states[0], 8), np.array([encode_action(a) for a in actions[:horizon]])
            Z_b = np.array([encode_state(s, 8) for s in box_states[1:horizon+1]])
            if len(oracle(SymbolicTrajectory.from_env_trajectory(box_states, actions))) > 0:
                test_pairs.append((z0_b, A_b, Z_b))

    return train_pairs, test_pairs


def generate_identity_pairs(num, horizon, seed):
    """Train: swap two keys. Test: swap key with box."""
    scenario = Scenario.from_preset("multi_object", 8)
    oracle = SymbolicOracle()
    rng = np.random.RandomState(seed)
    train_pairs, test_pairs = [], []

    for i in range(num):
        env = GridWorld(grid_size=8, objects_config=scenario.to_env_config(),
                       seed=int(rng.randint(0, 2**31)))
        env.reset()
        states, actions = [], []
        state = env.get_state()
        states.append(deepcopy(state))
        for t in range(horizon * 3):
            valid = env.get_valid_actions()
            if not valid: break
            a = int(rng.choice(valid))
            state, _, done, _ = env.step(a)
            states.append(deepcopy(state)); actions.append(a)
            if done: break
        if len(states) < horizon + 1: continue

        keys = [oid for oid in states[0].get("objects", {}).keys()
               if states[0]["objects"][oid].get("type") == "key"]
        boxes = [oid for oid in states[0].get("objects", {}).keys()
                if states[0]["objects"][oid].get("type") == "box"]

        # Train: swap two keys
        if len(keys) >= 2:
            k1, k2 = keys[0], keys[1]
            ks = deepcopy(states)
            t_swap = rng.randint(1, len(ks)-2)
            for t in range(t_swap, len(ks)):
                if k1 in ks[t]["objects"] and k2 in ks[t]["objects"]:
                    ks[t]["objects"][k1]["pos"], ks[t]["objects"][k2]["pos"] = \
                        ks[t]["objects"][k2]["pos"], ks[t]["objects"][k1]["pos"]
            z0, A = encode_state(ks[0], 8), np.array([encode_action(a) for a in actions[:horizon]])
            Z = np.array([encode_state(s, 8) for s in ks[1:horizon+1]])
            train_pairs.append((z0, A, Z))

        # Test: swap key with box
        if keys and boxes:
            k, b = keys[0], boxes[0]
            bs = deepcopy(states)
            t_swap = rng.randint(1, len(bs)-2)
            for t in range(t_swap, len(bs)):
                if k in bs[t]["objects"] and b in bs[t]["objects"]:
                    bs[t]["objects"][k]["pos"], bs[t]["objects"][b]["pos"] = \
                        bs[t]["objects"][b]["pos"], bs[t]["objects"][k]["pos"]
            z0_b, A_b = encode_state(bs[0], 8), np.array([encode_action(a) for a in actions[:horizon]])
            Z_b = np.array([encode_state(s, 8) for s in bs[1:horizon+1]])
            test_pairs.append((z0_b, A_b, Z_b))

    return train_pairs, test_pairs


def generate_locality_pairs(num, horizon, seed):
    """Train: push affects distant box. Test: pickup changes unrelated door."""
    scenario = Scenario.from_preset("multi_object", 8)
    rng = np.random.RandomState(seed)
    train_pairs, test_pairs = [], []

    for i in range(num):
        env = GridWorld(grid_size=8, objects_config=scenario.to_env_config(),
                       seed=int(rng.randint(0, 2**31)))
        env.reset()
        states, actions = [], []
        state = env.get_state()
        states.append(deepcopy(state))
        for t in range(horizon * 3):
            valid = env.get_valid_actions()
            if not valid: break
            a = int(rng.choice(valid))
            state, _, done, _ = env.step(a)
            states.append(deepcopy(state)); actions.append(a)
            if done: break
        if len(states) < horizon + 1: continue

        boxes = [oid for oid in states[0].get("objects", {}).keys()
                if states[0]["objects"][oid].get("type") == "box"]

        # Train: move a distant box
        if boxes:
            b = rng.choice(boxes)
            ts = deepcopy(states)
            t_loc = rng.randint(1, len(ts)-2)
            for t in range(t_loc, len(ts)):
                if b in ts[t]["objects"]:
                    old = ts[t]["objects"][b].get("pos", (0,0))
                    nr = min(7, max(0, old[0]+rng.choice([-1,1])))
                    nc = min(7, max(0, old[1]+rng.choice([-1,1])))
                    ts[t]["objects"][b]["pos"] = (nr, nc)
            z0_l, A_l = encode_state(ts[0], 8), np.array([encode_action(a) for a in actions[:horizon]])
            Z_l = np.array([encode_state(s, 8) for s in ts[1:horizon+1]])
            train_pairs.append((z0_l, A_l, Z_l))

        # Test: toggle door state
        doors = [oid for oid in states[0].get("objects", {}).keys()
                if states[0]["objects"][oid].get("type") == "door"]
        if doors:
            d = rng.choice(doors)
            ts2 = deepcopy(states)
            for t in range(1, len(ts2)):
                if d in ts2[t].get("door_states", {}):
                    ts2[t]["door_states"][d] = not ts2[t-1].get("door_states", {}).get(d, False)
            z0_d, A_d = encode_state(ts2[0], 8), np.array([encode_action(a) for a in actions[:horizon]])
            Z_d = np.array([encode_state(s, 8) for s in ts2[1:horizon+1]])
            test_pairs.append((z0_d, A_d, Z_d))

    return train_pairs, test_pairs


def generate_temporal_pairs(num, horizon, seed):
    """Train: reverse trajectory (strong temporal violation). Test: swap two distant states."""
    scenario = Scenario.from_preset("key_door_simple", 8)
    rng = np.random.RandomState(seed)
    train_pairs, test_pairs = [], []

    for i in range(num):
        env = GridWorld(grid_size=8, objects_config=scenario.to_env_config(),
                       seed=int(rng.randint(0, 2**31)))
        env.reset()
        states, actions = [], []
        state = env.get_state()
        states.append(deepcopy(state))
        for t in range(horizon * 3):
            valid = env.get_valid_actions()
            if not valid: break
            a = int(rng.choice(valid))
            state, _, done, _ = env.step(a)
            states.append(deepcopy(state)); actions.append(a)
            if done: break
        if len(states) < horizon + 1: continue

        # Train: reverse the state sequence (cause happens after effect)
        ts = deepcopy(states)
        ts[1:horizon+1] = list(reversed(ts[1:horizon+1]))
        act_rev = list(reversed(actions[:horizon]))
        z0, A = encode_state(ts[0], 8), np.array([encode_action(a) for a in act_rev])
        Z = np.array([encode_state(s, 8) for s in ts[1:horizon+1]])
        train_pairs.append((z0, A, Z))

        # Test: swap two non-adjacent states (subtler but still temporal)
        ts2 = deepcopy(states)
        if len(ts2) >= horizon:
            t1 = rng.randint(1, horizon//3)
            t2 = rng.randint(2*horizon//3, horizon)
            ts2[t1], ts2[t2] = ts2[t2], ts2[t1]
            acts2 = list(actions)
            if t1 < len(acts2) and t2 < len(acts2):
                acts2[t1], acts2[t2] = acts2[t2], acts2[t1]
            z0_t, A_t = encode_state(ts2[0], 8), np.array([encode_action(a) for a in acts2[:horizon]])
            Z_t = np.array([encode_state(s, 8) for s in ts2[1:horizon+1]])
            test_pairs.append((z0_t, A_t, Z_t))

    return train_pairs, test_pairs


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--num", type=int, default=500)
    parser.add_argument("--horizon", type=int, default=25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default="data/corruption_train.pkl")
    args = parser.parse_args()

    print("Generating explicit corruption training pairs...")
    valid = generate_valid(Scenario.from_preset("multi_object", 8), args.horizon, args.num, args.seed)
    print(f"Valid trajectories: {len(valid)}")

    cons_train, cons_test = generate_conservation_pairs(args.num, args.horizon, args.seed+1)
    print(f"Conservation: train={len(cons_train)}, test={len(cons_test)}")

    id_train, id_test = generate_identity_pairs(args.num, args.horizon, args.seed+2)
    print(f"Identity: train={len(id_train)}, test={len(id_test)}")

    loc_train, loc_test = generate_locality_pairs(args.num, args.horizon, args.seed+3)
    print(f"Locality: train={len(loc_train)}, test={len(loc_test)}")

    temp_train, temp_test = generate_temporal_pairs(args.num, args.horizon, args.seed+4)
    print(f"Temporal: train={len(temp_train)}, test={len(temp_test)}")

    data = {
        "valid": valid,
        "corruptions": {
            "conservation": {"train": cons_train, "test": cons_test},
            "identity": {"train": id_train, "test": id_test},
            "locality": {"train": loc_train, "test": loc_test},
            "temporal": {"train": temp_train, "test": temp_test},
        },
    }

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "wb") as f:
        pickle.dump(data, f)
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
