#!/usr/bin/env python3
"""Flat energy test: does IWCM energy on TAMG slots stay bounded as H increases?

Tests the drift elimination claim: solver-on-encoded-slots should produce
flat energy regardless of horizon. Replicates the H=10,25,50,100 test
from the paper but using TAMG-encoded pixel slots instead of oracle slots.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
import torch.nn.functional as F

from src.env.grid_world import GridWorld
from src.env.scenarios import Scenario
from src.env.renderer import GridWorldRenderer
from src.tamg.slot_encoder import TAMGSlotEncoder
from src.iwcm.fused_energy import FusedIWCMEnergy

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_SLOTS = 8


def load_tamg_model(checkpoint="outputs/tamg_encoder_best.pt"):
    """Load trained TAMG model (encoder + energy)."""
    from scripts.train_tamg_encoder import TAMGTrainer
    model = TAMGTrainer(num_slots=NUM_SLOTS, d_feat=64, img_size=64, hidden=128).to(DEVICE)
    model.load_state_dict(torch.load(checkpoint, map_location=DEVICE, weights_only=True))
    model.eval()
    return model


def encode_to_slots(model_enc, frames_np):
    """Frames → TAMG slots. frames_np: (1, H+1, 256, 256, 3)."""
    frames_t = torch.from_numpy(frames_np).float().to(DEVICE) / 255.0
    B, T = frames_t.shape[:2]
    frames_t = frames_t.permute(0, 1, 4, 2, 3)
    if frames_t.shape[-1] != 64:
        frames_t = F.interpolate(
            frames_t.reshape(-1, 3, *frames_t.shape[-2:]),
            size=(64, 64), mode='bilinear', align_corners=False
        ).reshape(B, T, 3, 64, 64)
    H = T - 1
    all_slots = []
    with torch.no_grad():
        for t in range(H):
            slots_t, _ = model_enc(frames_t[:, t], frames_t[:, t + 1])
            all_slots.append(slots_t)
    return torch.stack(all_slots, dim=1), all_slots[0]


def generate_trajectory(horizon, rng, scenario_name="key_door_simple"):
    """Generate a valid trajectory + rendered frames."""
    sc = Scenario.from_preset(scenario_name, 8)
    env = GridWorld(grid_size=8, objects_config=sc.to_env_config(),
                    seed=int(rng.randint(0, 2**31)))
    renderer = GridWorldRenderer(grid_size=8, cell_px=32, show_grid=False)
    env.reset()
    states, actions = [], []
    s = env.get_state(); states.append(s)
    for _ in range(horizon * 3):
        va = env.get_valid_actions()
        if not va: break
        a = int(rng.choice(va))
        s, _, done, _ = env.step(a)
        states.append(s); actions.append(a)
        if done: break
    if len(states) < horizon + 1 or len(actions) < horizon:
        return None
    frames = np.stack([renderer.render_frame(s) for s in states[:horizon + 1]], axis=0)
    return frames[None, :], np.array(actions[:horizon])


def run_drift_test(model, horizons=[10, 25, 50, 100], n_each=50):
    """Energy on TAMG slots across horizons. Also tests solver initialization."""
    results = {h: {"energy": [], "solver_energy": [], "rand_energy": []} for h in horizons}
    rng = np.random.RandomState(42)

    for H in horizons:
        print(f"Horizon H={H}... ", end="", flush=True)
        n_ok = 0
        while n_ok < n_each:
            result = generate_trajectory(H, rng)
            if result is None:
                continue
            frames_np, actions = result
            actions_onehot = np.zeros((1, H, 11), dtype=np.float32)
            for t, a in enumerate(actions):
                if 0 <= a < 11:
                    actions_onehot[0, t, a] = 1.0

            with torch.no_grad():
                Z, z0 = encode_to_slots(model.encoder, frames_np)
                A_t = torch.from_numpy(actions_onehot).float().to(DEVICE)
                e_tamg = model.energy_fn(z0, A_t, Z).item()
                Z_rand = torch.randn_like(Z)
                e_rand = model.energy_fn(z0, A_t, Z_rand).item()

            # Solver OUTSIDE no_grad so gradients flow
            Z_solv = Z_rand.clone().detach().requires_grad_(True)
            optim = torch.optim.SGD([Z_solv], lr=0.05)
            for _ in range(50):
                e = model.energy_fn(z0, A_t, Z_solv)
                optim.zero_grad()
                e.backward()
                optim.step()
                with torch.no_grad():
                    Z_solv.clamp_(-2, 2)
            e_solv = model.energy_fn(z0, A_t, Z_solv.detach()).item()

            results[H]["energy"].append(e_tamg)
            results[H]["solver_energy"].append(e_solv)
            results[H]["rand_energy"].append(e_rand)
            n_ok += 1

        print(f"TAMG: {np.mean(results[H]['energy']):.3f} ± {np.std(results[H]['energy']):.3f}  "
              f"Solver: {np.mean(results[H]['solver_energy']):.3f} ± {np.std(results[H]['solver_energy']):.3f}  "
              f"Random: {np.mean(results[H]['rand_energy']):.3f} ± {np.std(results[H]['rand_energy']):.3f}")

    print(f"\n{'='*70}")
    print("Drift analysis: does TAMG energy drift with H?")
    print(f"{'='*70}")
    print(f"{'H':>6} | {'TAMG energy':>12} | {'Solver energy':>14} | {'Random energy':>14} | {'Δsolver-TAMG':>14}")
    print("-" * 66)
    for H in horizons:
        e_t = np.mean(results[H]["energy"])
        e_s = np.mean(results[H]["solver_energy"])
        e_r = np.mean(results[H]["rand_energy"])
        delta = e_s - e_t
        print(f"{H:>6} | {e_t:>10.3f}  | {e_s:>10.3f}  | {e_r:>10.3f}  | {delta:>+10.3f}")

    # Flatness: slope of energy vs H
    Hs = np.array(horizons)
    energies = np.array([np.mean(results[H]["energy"]) for H in horizons])
    if len(horizons) >= 3:
        slope = np.polyfit(np.log(Hs), energies, 1)[0]
        print(f"\nEnergy vs log(H) slope: {slope:.4f}  (0 = flat)")
        print(f"Drift classification: {'FLAT ✓' if abs(slope) < 0.5 else 'DRIFTING ✗'}")
    else:
        print("\nNeed ≥3 horizons for drift slope analysis.")

    return results


def verify_eval_integrity():
    """Verify the evaluation fix: corruptions are applied to TAMG slots, not oracle Z."""
    H = 15
    rng = np.random.RandomState(42)
    result = generate_trajectory(H, rng)
    if result is None:
        return
    frames_np, actions = result
    actions_onehot = np.zeros((1, H, 11), dtype=np.float32)
    for t, a in enumerate(actions[:H]):
        if 0 <= a < 11:
            actions_onehot[0, t, a] = 1.0

    model = load_tamg_model()
    model.encoder.eval()
    model.energy_fn.eval()

    with torch.no_grad():
        Z, z0 = encode_to_slots(model.encoder, frames_np)
        A_t = torch.from_numpy(actions_onehot).float().to(DEVICE)
        e_valid = model.energy_fn(z0, A_t, Z).item()
        print(f"\n{'='*70}")
        print("Evaluation integrity check")
        print(f"{'='*70}")

        # Corrupt TAMG-encoded slots (the correct method)
        from src.tamg.slot_encoder import corrupt_tamg_slots
        Zc = corrupt_tamg_slots(Z.cpu().numpy(), np.random.RandomState(7))
        Zc_t = torch.from_numpy(Zc).float().to(DEVICE)
        e_corr_tamg = model.energy_fn(z0, A_t, Zc_t).item()

        print(f"  Valid TAMG slots:   {e_valid:.4f}")
        print(f"  Corrupted TAMG:     {e_corr_tamg:.4f}")
        print(f"  Energy gap:         {e_corr_tamg - e_valid:.4f}")

        # Verify TAMG slots and corrupted TAMG slots are different
        diff = (Zc_t != Z.to(DEVICE)).any().item()
        print(f"  Slots actually differ: {diff}")
        print(f"  → Eval is clean: {'YES ✓' if diff and e_corr_tamg > e_valid else 'ISSUE ✗'}")


if __name__ == "__main__":
    print(f"Device: {DEVICE}")
    print("=" * 70)
    print("Flat energy / drift test: TAMG slots across horizon")
    print("=" * 70)

    model = load_tamg_model()
    print(f"Model loaded: {sum(p.numel() for p in model.parameters()):,} params")

    run_drift_test(model, horizons=[10, 25, 50, 100], n_each=50)
    verify_eval_integrity()
