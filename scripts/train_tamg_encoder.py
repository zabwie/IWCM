#!/usr/bin/env python3
"""Train TAMGSlotEncoder + FusedIWCMEnergy jointly on pixel frames.

Pipeline: render grid world frames → TAMGSlotEncoder → (N,19) slots
→ FusedIWCMEnergy energy + self-supervised losses → backprop.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict
from sklearn.metrics import roc_auc_score

from src.utils.seed import set_seed
from src.env.grid_world import GridWorld
from src.env.scenarios import Scenario
from src.env.renderer import GridWorldRenderer
from src.tamg.slot_encoder import TAMGSlotEncoder, corrupt_tamg_slots
from src.iwcm.fused_energy import FusedIWCMEnergy
from src.env.oracle_slot_encoder import encode_oracle_trajectory, build_door_key_map, ORACLE_SLOT_DIM

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE}")


def generate_frame_data(n_trajectories=200, horizon=25, seed=42):
    """Generate trajectories + rendered frames + oracle slots.

    Returns per trajectory: (frames, z0_oracle, A_oracle, Z_oracle).
    frames: (H+1, 64, 64, 3) numpy uint8,  H+1 frames for H actions.
    """
    rng = np.random.RandomState(seed)
    renderer = GridWorldRenderer(grid_size=8, cell_px=32, show_grid=False)
    scenarios = {name: Scenario.from_preset(name, 8) for name in
                 ["key_door_simple", "multi_object", "conservation_test",
                  "counterfactual_test", "box_push", "splice_test"]}

    data = []
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

        # Oracle encoding (reference for initial state and actions)
        goal = states[0].get("goal", None)
        dkm = build_door_key_map(sc.to_env_config())
        oracle_result = encode_oracle_trajectory(
            states, actions, horizon, 8, goal, dkm)
        if oracle_result is None:
            continue
        z0_o, A_o, Z_o = oracle_result

        # Render H+1 frames
        frames_np = np.stack([
            renderer.render_frame(s) for s in states[:horizon + 1]
        ], axis=0)  # (H+1, 64, 64, 3)

        data.append((frames_np, z0_o, A_o, Z_o))
        n_gen += 1

    return data


class TAMGTrainer(nn.Module):
    """Wraps TAMGSlotEncoder + FusedIWCMEnergy for joint training."""

    def __init__(self, num_slots=8, d_feat=64, img_size=64, hidden=128):
        super().__init__()
        self.encoder = TAMGSlotEncoder(
            num_slots=num_slots, d_feat=d_feat, img_size=img_size)
        self.energy_fn = FusedIWCMEnergy(
            d_slot=19, d_action=11, hidden=hidden, num_slots=num_slots)

    def forward(self, frames, z0_o, A_o, Z_o, rng, margin=1.0, reg=0.001):
        """Joint forward: encode frames → slots → energy + self-supervised losses.

        Args:
            frames: (B, H+1, 64, 64, 3) numpy uint8.
            z0_o: (B, N, 19) oracle initial state (for z0).
            A_o: (B, H, 11) oracle actions.
            Z_o: (B, H, N, 19) oracle slots (for corruption detection).
        Returns:
            total_loss, debug dict.
        """
        B, Hp1 = frames.shape[:2]
        H = Hp1 - 1  # H frame pairs → H slot timesteps

        frames_t = _preprocess_frames(frames)
        Z, z0, all_losses = _encode_to_slots(self.encoder, frames_t)

        # ── IWCM margin loss ────────────────────────────────────────────
        A = torch.from_numpy(A_o).float().to(DEVICE)  # (B, H, 11)

        # Corrupt Z using tamg-aware corruption
        Z_np = Z.detach().cpu().numpy()
        Zc_np = corrupt_tamg_slots(Z_np, rng)
        Zc = torch.from_numpy(Zc_np).float().to(DEVICE)

        # Ensure corruption actually changed something
        diff = (Zc != Z).any(dim=(-1, -2, -3))
        if diff.sum() < 2:
            Ev = self.energy_fn(z0, A, Z)
            loss_margin = F.relu(Ev + 0.5).mean() + reg * Ev.pow(2).mean()
        else:
            z0_f, A_f = z0[diff], A[diff]
            Zv_f, Zc_f = Z[diff], Zc[diff]
            Ev = self.energy_fn(z0_f, A_f, Zv_f)
            Ec = self.energy_fn(z0_f, A_f, Zc_f)
            loss_margin = F.relu(Ev + 0.5).mean()
            loss_margin = loss_margin + F.relu(Ev + margin - Ec).mean()
            loss_margin = loss_margin + reg * (Ev.pow(2).mean() + Ec.pow(2).mean())

        # ── Self-supervised losses ──────────────────────────────────────
        loss_vel = torch.stack(all_losses['loss_vel']).mean()
        loss_id = torch.stack(all_losses['loss_id']).mean()
        loss_type = torch.stack(all_losses['loss_type']).mean()
        loss_recon = torch.stack(all_losses['loss_recon']).mean()

        # ── Total ───────────────────────────────────────────────────────
        total = (loss_margin
                 + 0.1 * loss_vel
                 + 0.1 * loss_id
                 + 0.05 * loss_type
                 + 0.1 * loss_recon)

        debug = {
            'loss_total': total.item(),
            'loss_margin': loss_margin.item(),
            'loss_vel': loss_vel.item(),
            'loss_id': loss_id.item(),
            'loss_type': loss_type.item(),
            'loss_recon': loss_recon.item(),
            'energy_valid': Ev.mean().item() if diff.sum() >= 2 else Ev.mean().item(),
        }
        if diff.sum() >= 2:
            debug['energy_corrupt'] = Ec.mean().item()
            debug['energy_gap'] = (Ec - Ev).mean().item()

        return total, debug


def _preprocess_frames(frames_np):
    """Normalize, permute, resize to 64×64."""
    frames_t = torch.from_numpy(frames_np).float().to(DEVICE) / 255.0
    if frames_t.dim() == 4:
        frames_t = frames_t.unsqueeze(0)
    frames_t = frames_t.permute(0, 1, 4, 2, 3)  # (B, T, 3, H, W)
    if frames_t.shape[-1] != 64:
        B, T = frames_t.shape[:2]
        frames_t = F.interpolate(
            frames_t.reshape(-1, 3, *frames_t.shape[-2:]),
            size=(64, 64), mode='bilinear', align_corners=False
        ).reshape(B, T, 3, 64, 64)
    return frames_t


def _encode_to_slots(model_enc, frames_t):
    """Encode frame sequence to (B, H, N, 19) slots + loss dicts."""
    B, Hp1 = frames_t.shape[:2]
    H = Hp1 - 1
    all_slots, all_losses = [], defaultdict(list)
    for t in range(H):
        slots_t, losses_t = model_enc(frames_t[:, t], frames_t[:, t + 1])
        all_slots.append(slots_t)
        for k, v in losses_t.items():
            all_losses[k].append(v)
    return torch.stack(all_slots, dim=1), all_slots[0], all_losses


def evaluate(model, valid_data, corr_data, oracle_ref=False):
    """AUROC: encode TAMG slots, corrupt them on-the-fly, compare energies."""
    rng_eval = np.random.RandomState(42)
    model.eval()
    valid_energies = []
    for frames, z0_o, A_o, Z_o in valid_data:
        frames_t = _preprocess_frames(frames)
        with torch.no_grad():
            Z, z0, _ = _encode_to_slots(model.encoder, frames_t)
            A = torch.from_numpy(A_o).float().to(DEVICE)
            e = model.energy_fn(z0, A, Z).item()
        valid_energies.append(e)

    corr_energies = defaultdict(list)
    for frames, z0_o, A_o, Z_o, meta in corr_data:
        frames_t = _preprocess_frames(frames)
        with torch.no_grad():
            Z, z0, _ = _encode_to_slots(model.encoder, frames_t)
            # Corrupt the TAMG-encoded slots, not the oracle Z
            Zc_np = corrupt_tamg_slots(Z.cpu().numpy(), rng_eval)
            Zc = torch.from_numpy(Zc_np).float().to(DEVICE)
            A = torch.from_numpy(A_o).float().to(DEVICE)
            e = model.energy_fn(z0, A, Zc).item()
        vtype = meta.get("violation_type", meta.get("vtype", "unknown"))
        corr_energies[vtype].append(e)

    results = {}
    for vtype, ces in corr_energies.items():
        labels = [0] * len(valid_energies) + [1] * len(ces)
        scores = valid_energies + ces
        if len(set(labels)) < 2 or len(set(scores)) < 2:
            continue
        results[vtype] = roc_auc_score(labels, scores)

    all_corr = [e for es in corr_energies.values() for e in es]
    if len(set([0] * len(valid_energies) + [1] * len(all_corr))) >= 2 and len(set(valid_energies + all_corr)) >= 2:
        results["overall"] = roc_auc_score(
            [0] * len(valid_energies) + [1] * len(all_corr),
            valid_energies + all_corr
        )
    return results


def generate_corruptions(data_list, rng, n_per=5):
    """Mark which trajectories to corrupt for evaluation."""
    corrupted = []
    for frames, z0_o, A_o, Z_o in data_list:
        corrupted.append((frames, z0_o, A_o, Z_o, {"vtype": "mechanical"}))
    return corrupted[:len(data_list) * n_per]


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-traj", type=int, default=200)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--horizon", type=int, default=15)
    args = parser.parse_args()

    print("=" * 70)
    print("TAMGSlotEncoder + FusedIWCMEnergy joint training")
    print(f"Trajectories: {args.n_traj}  Horizon: {args.horizon}  Epochs: {args.epochs}")
    print("=" * 70)

    set_seed(args.seed)
    rng = np.random.RandomState(args.seed)

    # ── Generate data ───────────────────────────────────────────────────
    print("\nGenerating trajectories and rendering frames...")
    data = generate_frame_data(args.n_traj, args.horizon, args.seed)
    n_train = int(len(data) * 0.6)
    n_val = int(len(data) * 0.2)
    train_data = data[:n_train]
    val_data = data[n_train:n_train + n_val]
    test_data = data[n_train + n_val:]
    print(f"  Train: {len(train_data)}  Val: {len(val_data)}  Test: {len(test_data)}")

    # Generate corrupted data for validation
    print("Generating corrupted trajectories...")
    val_corr = generate_corruptions(val_data, np.random.RandomState(args.seed + 1))
    print(f"  Validation corruptions: {len(val_corr)}")

    # ── Build model ─────────────────────────────────────────────────────
    model = TAMGTrainer(num_slots=8, d_feat=64, img_size=64, hidden=128).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel parameters: {n_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # ── Training loop ───────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"Training")
    print(f"{'='*70}")

    best_auc = 0.0
    for epoch in range(args.epochs):
        model.train()
        rng_epoch = np.random.RandomState(args.seed + epoch)
        epoch_losses = defaultdict(list)

        # Shuffle
        idx = np.random.permutation(len(train_data))
        for start in range(0, len(train_data), args.batch_size):
            batch_idx = idx[start:start + args.batch_size]
            batch = [train_data[i] for i in batch_idx]

            # Stack batch
            frames = np.stack([b[0] for b in batch], axis=0)
            z0_o = np.stack([b[1] for b in batch], axis=0)
            A_o = np.stack([b[2] for b in batch], axis=0)
            Z_o = np.stack([b[3] for b in batch], axis=0)

            optimizer.zero_grad()
            loss, debug = model(frames, z0_o, A_o, Z_o, rng_epoch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            for k, v in debug.items():
                epoch_losses[k].append(v)

        # Log
        avg = {k: np.mean(v) for k, v in epoch_losses.items()}
        msg = f"Epoch {epoch:3d} | total: {avg['loss_total']:.4f} | margin: {avg['loss_margin']:.4f} | vel: {avg['loss_vel']:.4f} | id: {avg['loss_id']:.4f} | type: {avg['loss_type']:.4f} | recon: {avg['loss_recon']:.4f}"
        if 'energy_gap' in avg:
            msg += f" | gap: {avg['energy_gap']:.4f}"
        print(msg)

        # Evaluate every 10 epochs
        if (epoch + 1) % 10 == 0 or epoch == 0:
            results = evaluate(model, val_data, val_corr)
            auc = results.get("overall", 0)
            print(f"  Val AUROC: {auc:.4f}  (best: {max(best_auc, auc):.4f})")
            if auc > best_auc:
                best_auc = auc
                torch.save(model.state_dict(), "outputs/tamg_encoder_best.pt")
                print(f"  Saved best model (AUROC={auc:.4f})")

    print(f"\nBest validation AUROC: {best_auc:.4f}")

    # ── Final evaluation ────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("Final evaluation on test set")
    print(f"{'='*70}")

    # Load best
    model.load_state_dict(torch.load("outputs/tamg_encoder_best.pt", weights_only=True))

    test_corr = generate_corruptions(test_data, np.random.RandomState(args.seed + 2))
    results = evaluate(model, test_data, test_corr)
    for vtype, auc in sorted(results.items()):
        marker = " *" if vtype == "overall" else ""
        print(f"  {vtype:<20s}: {auc:.4f}{marker}")

    print("\nDone.")


if __name__ == "__main__":
    main()
