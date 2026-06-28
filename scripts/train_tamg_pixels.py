#!/usr/bin/env python3
"""End-to-end pixel pipeline: frames → slots → SimpleTAMG → causal detection.

Generates valid trajectories from the grid world, renders them as pixel
frames, extracts object slots via color segmentation and tracking, trains
SimpleTAMG on pixel-derived slots, and evaluates on compositional corruptions.

No oracle state access during slot extraction — only rendered pixel frames.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
from collections import defaultdict
from sklearn.metrics import roc_auc_score

from src.utils.seed import set_seed
from src.env.grid_world import GridWorld
from src.env.scenarios import Scenario, PREDEFINED_SCENARIOS
from src.env.renderer import GridWorldRenderer
from src.pixel_slots import extract_slots, SLOT_DIM as PIXEL_SLOT_DIM
from src.encoder.oracle_slot_encoder import encode_oracle_trajectory, build_door_key_map, ORACLE_SLOT_DIM, MAX_OBJECTS
from src.tamg_simple import SimpleTAMG, _corrupt
from src.iwcm.fused_energy import FusedIWCMEnergy

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def generate_pixel_data(n_trajectories=100, horizon=25, seed=42):
    """Generate valid trajectories, render frames, extract pixel + oracle slots."""
    rng = np.random.RandomState(seed)
    renderer = GridWorldRenderer(grid_size=8, cell_px=32, show_grid=False)
    scenarios = {name: Scenario.from_preset(name, 8) for name in
                 ["key_door_simple", "multi_object", "conservation_test",
                  "counterfactual_test", "box_push", "splice_test"]}

    pixel_slots_list = []
    oracle_slots_list = []
    n_gen = 0

    while n_gen < n_trajectories:
        sc = rng.choice(list(scenarios.values()))
        env = GridWorld(grid_size=8, objects_config=sc.to_env_config(),
                        seed=int(rng.randint(0, 2**31)))
        env.reset()
        states, actions = [], []
        state = env.get_state()
        states.append(state)
        for _ in range(horizon * 3):
            valid_acts = env.get_valid_actions()
            if not valid_acts:
                break
            a = int(rng.choice(valid_acts))
            state, _, done, _ = env.step(a)
            states.append(state)
            actions.append(a)
            if done:
                break

        if len(states) < horizon + 1 or len(actions) < horizon:
            continue

        # Oracle slot encoding (ground truth reference)
        goal = states[0].get("goal", None)
        dkm = build_door_key_map(sc.to_env_config())
        oracle_result = encode_oracle_trajectory(
            states, actions, horizon, 8, goal, dkm)
        if oracle_result is None:
            continue
        z0_o, A_o, Z_o = oracle_result

        # Render frames and extract pixel slots
        frames = [renderer.render_frame(s) for s in states[:horizon + 1]]
        Z_pixel = extract_slots(frames, grid_size=8)

        pixel_slots_list.append((z0_o, A_o, Z_pixel))
        oracle_slots_list.append((z0_o, A_o, Z_o))
        n_gen += 1

    return pixel_slots_list, oracle_slots_list


def evaluate_energy(model, test_valid, test_corr):
    model.eval()
    valid_scores = []
    for z0, A, Z in test_valid:
        s = model(z0.unsqueeze(0).to(DEVICE), A.unsqueeze(0).to(DEVICE),
                  Z.unsqueeze(0).to(DEVICE)).item()
        valid_scores.append(s)

    by_type = defaultdict(list)
    for z0, A, Z, meta in test_corr:
        by_type[meta["violation_type"]].append((z0, A, Z))

    results = {}
    for vtype, items in sorted(by_type.items()):
        cs = [model(z0.unsqueeze(0).to(DEVICE), A.unsqueeze(0).to(DEVICE),
                    Z.unsqueeze(0).to(DEVICE)).item() for z0, A, Z in items]
        results[vtype] = roc_auc_score([0] * len(valid_scores) + [1] * len(cs),
                                        valid_scores + cs)

    all_corr = [model(z0.unsqueeze(0).to(DEVICE), A.unsqueeze(0).to(DEVICE),
                      Z.unsqueeze(0).to(DEVICE)).item()
                for items in by_type.values() for z0, A, Z in items]
    results["overall"] = roc_auc_score(
        [0] * len(valid_scores) + [1] * len(all_corr),
        valid_scores + all_corr)
    return results


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_traj", type=int, default=200)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print("=" * 70)
    print("Pixel Pipeline: frames → slots → SimpleTAMG → causal detection")
    print(f"Trajectories: {args.n_traj}  Epochs: {args.epochs}")
    print("=" * 70)

    # ─── Generate data ──────────────────────────────────────────────────
    print("\nGenerating trajectories, rendering frames, extracting slots...")
    set_seed(args.seed)
    pixel_data, oracle_data = generate_pixel_data(args.n_traj, seed=args.seed)

    n_train = int(len(pixel_data) * 0.6)
    train_pixel = pixel_data[:n_train]
    test_pixel = pixel_data[n_train:]
    train_oracle = oracle_data[:n_train]

    print(f"  Generated {len(pixel_data)} trajectories")
    print(f"  Train: {n_train}  Test: {len(test_pixel)}")

    # ─── Pixel-slot SimpleTAMG ──────────────────────────────────────────
    print(f"\n{'='*70}")
    print("Training SimpleTAMG on PIXEL-DERIVED slots")
    print(f"{'='*70}")

    model_pixel = SimpleTAMG(d_slot=PIXEL_SLOT_DIM, hidden=128, num_slots=MAX_OBJECTS).to(DEVICE)
    opt_p = torch.optim.Adam(model_pixel.parameters(), lr=3e-3)
    nv = len(train_pixel)
    batch = 32
    all_p = [(torch.from_numpy(z0).float().to(DEVICE),
              torch.from_numpy(A).float().to(DEVICE),
              torch.from_numpy(Z).float().to(DEVICE))
             for z0, A, Z in train_pixel]

    for ep in range(args.epochs):
        for _ in range(max(1, nv // batch)):
            vi = np.random.choice(nv, min(batch, nv), replace=False)
            vz0 = torch.stack([all_p[i][0] for i in vi])
            vA = torch.stack([all_p[i][1] for i in vi])
            vZ = torch.stack([all_p[i][2] for i in vi])
            opt_p.zero_grad()
            loss = model_pixel.training_step(vz0, vA, vZ)
            if loss.item() == 0:
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model_pixel.parameters(), 1.0)
            opt_p.step()

    # Evaluate pixel-slot model on PIXEL test data
    test_pixel_valid = [(torch.from_numpy(z0).float(),
                         torch.from_numpy(A).float(),
                         torch.from_numpy(Z).float())
                        for z0, A, Z in test_pixel]

    # Generate corruptions mechanically on pixel test data
    test_pixel_corr = []
    rng = np.random.RandomState(args.seed + 99)
    for z0, A, Z in test_pixel_valid:
        Z_t = Z.unsqueeze(0)
        Zc = _corrupt(Z_t.clone(), rng)
        if (Zc != Z_t).any():
            test_pixel_corr.append((
                z0.clone(), A.clone(), Zc.squeeze(0).clone(),
                {"violation_type": "mechanical"}))

    results_pixel = evaluate_energy(model_pixel, test_pixel_valid, test_pixel_corr)
    print(f"\n  Pixel-slot SimpleTAMG (mechanical corruptions):")
    for vtype, auc in sorted(results_pixel.items()):
        marker = " *" if vtype == "overall" else ""
        print(f"    {vtype:<14s}: {auc:.4f}{marker}")

    # ─── Oracle-slot SimpleTAMG (baseline) ──────────────────────────────
    print(f"\n{'='*70}")
    print("Training SimpleTAMG on ORACLE slots (baseline)")
    print(f"{'='*70}")

    model_oracle = SimpleTAMG(d_slot=ORACLE_SLOT_DIM, hidden=128, num_slots=MAX_OBJECTS).to(DEVICE)
    opt_o = torch.optim.Adam(model_oracle.parameters(), lr=3e-3)
    all_o = [(torch.from_numpy(z0).float().to(DEVICE),
              torch.from_numpy(A).float().to(DEVICE),
              torch.from_numpy(Z).float().to(DEVICE))
             for z0, A, Z in train_oracle]

    for ep in range(args.epochs):
        for _ in range(max(1, nv // batch)):
            vi = np.random.choice(nv, min(batch, nv), replace=False)
            vz0 = torch.stack([all_o[i][0] for i in vi])
            vA = torch.stack([all_o[i][1] for i in vi])
            vZ = torch.stack([all_o[i][2] for i in vi])
            opt_o.zero_grad()
            loss = model_oracle.training_step(vz0, vA, vZ)
            if loss.item() == 0:
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model_oracle.parameters(), 1.0)
            opt_o.step()

    # Evaluate oracle-slot model on ORACLE test data
    test_o_valid = [(torch.from_numpy(z0).float(),
                     torch.from_numpy(A).float(),
                     torch.from_numpy(Z).float())
                    for z0, A, Z in test_pixel]
    test_o_corr = []
    rng2 = np.random.RandomState(args.seed + 99)
    for z0, A, Z in test_o_valid:
        Z_t = Z.unsqueeze(0)
        Zc = _corrupt(Z_t.clone(), rng2)
        if (Zc != Z_t).any():
            test_o_corr.append((
                z0.clone(), A.clone(), Zc.squeeze(0).clone(),
                {"violation_type": "mechanical"}))

    results_oracle = evaluate_energy(model_oracle, test_o_valid, test_o_corr)
    print(f"\n  Oracle-slot SimpleTAMG (mechanical corruptions):")
    for vtype, auc in sorted(results_oracle.items()):
        print(f"    {vtype:<14s}: {auc:.4f}")

    # ─── Cross-evaluate ─────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("CROSS-EVALUATION")
    print(f"{'='*70}")
    print(f"  Pixel model on oracle test data:")
    results_cross = evaluate_energy(model_pixel, test_o_valid, test_o_corr)
    for vtype, auc in sorted(results_cross.items()):
        marker = " *" if vtype == "overall" else ""
        print(f"    {vtype:<14s}: {auc:.4f}{marker}")

    print(f"\n  Oracle model on pixel test data:")
    results_cross2 = evaluate_energy(model_oracle, test_pixel_valid, test_pixel_corr)
    for vtype, auc in sorted(results_cross2.items()):
        marker = " *" if vtype == "overall" else ""
        print(f"    {vtype:<14s}: {auc:.4f}{marker}")

    p_ov = results_pixel.get("overall", 0)
    c_ov = results_cross.get("overall", 0)
    print(f"\n  Pixel→pixel: {p_ov:.4f}")
    print(f"  Pixel→oracle: {c_ov:.4f}")
    print(f"  Oracle→oracle: {results_oracle.get('overall', 0):.4f}")


if __name__ == "__main__":
    main()
