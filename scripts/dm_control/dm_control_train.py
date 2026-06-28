#!/usr/bin/env python3
"""DM Control IWCM experiment — causal law detection on physics-based environments.

Generates valid and corrupted trajectories from DM Control, encodes
them into oracle-structured slots (using physics state), and trains
a FusedIWCMEnergy to detect causal violations.

Since we use oracle-structured slot encoding (extracting per-body position,
velocity, and identity from MuJoCo physics), the IWCM should achieve high
AUROC — matching the grid world oracle performance (~0.93 AUROC).

Usage:
  python scripts/dm_control_train.py --domain cartpole --task swingup
  python scripts/dm_control_train.py --domain cheetah --task run
  python scripts/dm_control_train.py --domain walker --task walk
"""

import sys, time, argparse, numpy as np, torch, torch.nn.functional as F
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.seed import set_seed
from src.env.dm_control_wrapper import DMControlWrapper
from src.env.dm_control_encoder import (
    DMControlOracleEncoder, ORACLE_SLOT_DIM, MAX_BODIES, DOMAIN_CONFIGS,
)
from src.iwcm.fused_energy import FusedIWCMEnergy

HORIZON = 25
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def generate_dataset(wrapper, encoder, n_valid, n_corrupt, rng):
    """Generate valid and corrupted trajectory datasets.

    Uses raw MuJoCo physics state (qpos/qvel) for linear position/velocity
    encoding, avoiding cos/sin encoding that masks changes near vertical.

    Returns:
        valid_data: list of (z0, A, Z) for valid trajectories.
        corrupt_data: list of (z0, A, Z, meta) for corrupted trajectories.
    """
    valid_data = []
    corrupt_data = []

    corruption_types = ['teleport', 'freeze', 'reverse']
    corrupt_rng = np.random.RandomState(rng.randint(0, 2**31))

    for _ in range(n_valid):
        result = wrapper.generate_trajectory(horizon=HORIZON, random_policy=True)
        if result is None:
            continue
        physics_states, _, actions = result
        encoded = encoder.encode_trajectory(physics_states, actions, HORIZON)
        if encoded is None:
            continue
        z0, A, Z = encoded
        valid_data.append((
            torch.from_numpy(z0).float(),
            torch.from_numpy(A).float(),
            torch.from_numpy(Z).float(),
        ))

    attempts = 0
    max_attempts = n_corrupt * 5
    while len(corrupt_data) < n_corrupt and attempts < max_attempts:
        attempts += 1
        ct = corruption_types[corrupt_rng.randint(0, len(corruption_types))]
        result = wrapper.generate_corrupted_trajectory(
            horizon=HORIZON, corruption_type=ct, rng=corrupt_rng,
        )
        if result is None:
            continue
        physics_states, _, actions, meta = result
        encoded = encoder.encode_trajectory(physics_states, actions, HORIZON)
        if encoded is None:
            continue
        z0, A, Z = encoded
        corrupt_data.append((
            torch.from_numpy(z0).float(),
            torch.from_numpy(A).float(),
            torch.from_numpy(Z).float(),
            meta,
        ))

    if len(corrupt_data) < n_corrupt:
        print(f"  Warning: only got {len(corrupt_data)}/{n_corrupt} corrupt "
              f"trajectories after {attempts} attempts")

    return valid_data, corrupt_data


def train_iwcm(valid_data, corrupt_data, epochs=300, lr=1e-3, batch_size=32):
    """Train FusedIWCMEnergy on oracle-structured DM Control slots.

    This should work well because the oracle encoder produces structured
    slots with explicit position, velocity, and identity channels — the
    same format that the IWCM was validated on in grid world experiments.
    """
    d_slot = valid_data[0][2].shape[-1]  # 19
    d_action = valid_data[0][1].shape[-1]  # varies by domain
    num_slots = valid_data[0][2].shape[1]  # MAX_BODIES = 8

    model = FusedIWCMEnergy(
        d_slot=d_slot, d_action=d_action,
        hidden=128, num_slots=num_slots,
    ).to(DEVICE)

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    n = min(batch_size, len(valid_data), len(corrupt_data))

    history = {'ev': [], 'ec': [], 'loss': []}

    for ep in range(epochs):
        vi = np.random.choice(len(valid_data), n, replace=False)
        ci = np.random.choice(len(corrupt_data), n, replace=False)

        vz0 = torch.stack([valid_data[i][0] for i in vi]).to(DEVICE)
        vA = torch.stack([valid_data[i][1] for i in vi]).to(DEVICE)
        vZ = torch.stack([valid_data[i][2] for i in vi]).to(DEVICE)

        cz0 = torch.stack([corrupt_data[i][0] for i in ci]).to(DEVICE)
        cA = torch.stack([corrupt_data[i][1] for i in ci]).to(DEVICE)
        cZ = torch.stack([corrupt_data[i][2] for i in ci]).to(DEVICE)

        opt.zero_grad()
        ev = model(vz0, vA, vZ)
        ec = model(cz0, cA, cZ)

        loss = (F.relu(ev + 1.0).mean() +
                F.relu(1.0 - ec).mean() +
                0.001 * (ev.pow(2).mean() + ec.pow(2).mean()))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if (ep + 1) % 50 == 0:
            evm = ev.mean().item()
            ecm = ec.mean().item()
            history['ev'].append(evm)
            history['ec'].append(ecm)
            history['loss'].append(loss.item())
            print(f"  ep {ep+1:4d}: ev={evm:+.3f} ec={ecm:+.3f} loss={loss.item():.4f}")

    model.eval()
    return model, history


def evaluate(model, valid_data, corrupt_data):
    """Evaluate IWCM on held-out data."""
    energies, labels = [], []

    with torch.no_grad():
        for z0, A, Z in valid_data:
            e = model(
                z0.unsqueeze(0).to(DEVICE),
                A.unsqueeze(0).to(DEVICE),
                Z.unsqueeze(0).to(DEVICE),
            ).item()
            energies.append(e)
            labels.append(0)

        for z0, A, Z, _ in corrupt_data:
            e = model(
                z0.unsqueeze(0).to(DEVICE),
                A.unsqueeze(0).to(DEVICE),
                Z.unsqueeze(0).to(DEVICE),
            ).item()
            energies.append(e)
            labels.append(1)

    energies = np.array(energies)
    labels = np.array(labels)

    from sklearn.metrics import roc_auc_score, average_precision_score
    auroc = roc_auc_score(labels, energies)
    auprc = average_precision_score(labels, energies)

    ev = energies[labels == 0]
    ei = energies[labels == 1]
    overlap = (ev > ei.min()).mean() if len(ei) > 0 else 1.0

    return {
        'AUROC': auroc,
        'AUPRC': auprc,
        'E_valid_mean': float(ev.mean()),
        'E_invalid_mean': float(ei.mean()),
        'E_valid_std': float(ev.std()),
        'E_invalid_std': float(ei.std()),
        'overlap': float(overlap),
        'n_valid': len(ev),
        'n_invalid': len(ei),
    }


def main():
    parser = argparse.ArgumentParser(description='DM Control IWCM experiment')
    parser.add_argument('--domain', default='cartpole', help='DM Control domain')
    parser.add_argument('--task', default='swingup', help='Task within domain')
    parser.add_argument('--horizon', type=int, default=HORIZON, help='Trajectory horizon')
    parser.add_argument('--n_train', type=int, default=200, help='Training trajectories per class')
    parser.add_argument('--n_test', type=int, default=50, help='Test trajectories per class')
    parser.add_argument('--epochs', type=int, default=300, help='Training epochs')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--no_render', action='store_true', help='Skip frame rendering (faster)')
    args = parser.parse_args()

    set_seed(args.seed)
    rng = np.random.RandomState(args.seed)

    print(f"DM Control IWCM — {args.domain}/{args.task}")
    print("=" * 60)

    # Environment
    t0 = time.time()
    wrapper = DMControlWrapper(
        args.domain, args.task, seed=args.seed,
        render_size=(64, 64) if not args.no_render else (16, 16),
    )
    encoder = DMControlOracleEncoder(args.domain, max_bodies=MAX_BODIES)
    print(f"State dim: {wrapper.state_dim}, Action dim: {wrapper.action_dim}")
    print(f"Bodies: {[b['name'] for b in encoder.config['bodies']]}")
    print(f"Encoder slot dim: {ORACLE_SLOT_DIM}, max bodies: {MAX_BODIES}")

    # Generate data
    print(f"\nGenerating {args.n_train} valid + {args.n_train} corrupt trajectories...")
    train_valid, train_corrupt = generate_dataset(
        wrapper, encoder, args.n_train, args.n_train, rng,
    )
    print(f"  Train: {len(train_valid)} valid, {len(train_corrupt)} corrupt ({time.time() - t0:.1f}s)")

    t0 = time.time()
    test_valid, test_corrupt = generate_dataset(
        wrapper, encoder, args.n_test, args.n_test, np.random.RandomState(args.seed + 1),
    )
    print(f"  Test:  {len(test_valid)} valid, {len(test_corrupt)} corrupt ({time.time() - t0:.1f}s)")

    # Train
    print(f"\nTraining IWCM ({args.epochs} epochs)...")
    t0 = time.time()
    model, history = train_iwcm(
        train_valid, train_corrupt,
        epochs=args.epochs, lr=args.lr,
    )
    print(f"  Done ({time.time() - t0:.1f}s)")

    # Evaluate
    print("\n" + "=" * 60)
    train_metrics = evaluate(model, train_valid, train_corrupt)
    test_metrics = evaluate(model, test_valid, test_corrupt)

    for name, m in [("Train", train_metrics), ("Test", test_metrics)]:
        print(f"\n{name}:")
        print(f"  AUROC:           {m['AUROC']:.3f}")
        print(f"  AUPRC:           {m['AUPRC']:.3f}")
        print(f"  E_valid:         {m['E_valid_mean']:+.3f} ± {m['E_valid_std']:.3f}")
        print(f"  E_invalid:       {m['E_invalid_mean']:+.3f} ± {m['E_invalid_std']:.3f}")
        print(f"  Energy margin:   {m['E_invalid_mean'] - m['E_valid_mean']:+.3f}")
        print(f"  Overlap:         {m['overlap']:.3f}")

    # Verdict
    print("\n" + "=" * 60)
    auroc = test_metrics['AUROC']
    if auroc >= 0.85:
        print(f"PASS — AUROC {auroc:.3f} meets threshold for {args.domain}/{args.task}")
    elif auroc >= 0.75:
        print(f"PROMISING — AUROC {auroc:.3f}. Increase epochs or data for {args.domain}/{args.task}")
    elif auroc >= 0.65:
        print(f"MODERATE — AUROC {auroc:.3f}. Encoder may need richer features for {args.domain}")
    else:
        print(f"WEAK — AUROC {auroc:.3f}. Check encoder configuration for {args.domain}")


if __name__ == "__main__":
    main()
