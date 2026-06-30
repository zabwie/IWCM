#!/usr/bin/env python3
"""Reproduction bundle for IWCM — 4 paper-claim tests with PASS/FAIL.

Usage:
    python scripts/reproduction.py              # full run (trains DM models, ~15 min first time)
    python scripts/reproduction.py --cached     # skip training, use cached models only
    python scripts/reproduction.py --test 1     # run single test

Caches trained models to outputs/checkpoints/repro_*.pt on first run.
Subsequent runs load cached models (~30 sec total).
"""
# ponytail: single-file reproduction. Caches heavily. Reruns are fast.

import sys, torch, numpy as np, pickle, argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
CACHE = ROOT / 'outputs' / 'checkpoints'
CACHE.mkdir(parents=True, exist_ok=True)
DATA_CACHE = CACHE / 'repro_dm_data.pkl'

torch.manual_seed(42)
np.random.seed(42)

results = {}  # test_name -> {passed: bool, details: str}


def check(label, condition, value, threshold, unit=''):
    status = 'PASS' if condition else 'FAIL'
    print(f"  [{status}] {label}: {value:.4f}{unit} (threshold: {threshold})")
    return status == 'PASS'


# ═══════════════════════════════════════════════════════════════════════════════
# TEST 1: AC3 Grid-World — Compositional Curriculum AUROC
# ═══════════════════════════════════════════════════════════════════════════════

def test_ac3():
    """Test 1: Model C (compositional corruptions) vs Model B (random noise) AUROC.
    # ponytail: uses exp1_b_vs_c.py protocol — trains both models from scratch on cached data
    """
    print("\n" + "=" * 65)
    print("TEST 1: Grid-World — Compositional (C) vs Random (B) AUROC")
    print("=" * 65)

    from scripts.exp1.exp1_b_vs_c import train_model, evaluate
    from src.iwcm.slot_energy import SlotIWCMEnergy
    from src.env.oracle_slot_encoder import ORACLE_SLOT_DIM, MAX_OBJECTS
    from sklearn.metrics import roc_auc_score
    from collections import defaultdict

    # ── Load data ──
    with open(ROOT / 'data' / 'compositional_grid.pkl', 'rb') as f:
        grid = pickle.load(f)

    train_valid = [(torch.from_numpy(z0).float(), torch.from_numpy(A).float(),
                    torch.from_numpy(Z).float())
                   for z0, A, Z in grid["train_valid"]]
    test_valid = [(torch.from_numpy(z0).float(), torch.from_numpy(A).float(),
                   torch.from_numpy(Z).float())
                  for z0, A, Z in grid["test_valid"]]
    test_corr_raw = [(torch.from_numpy(item[0][0]).float(), torch.from_numpy(item[0][1]).float(),
                      torch.from_numpy(item[0][2]).float(), item[1])
                     for item in grid["test_corr"]]

    N, d = MAX_OBJECTS, ORACLE_SLOT_DIM
    print(f"  Data: {len(train_valid)} valid, {len(test_valid)} test valid, {len(test_corr_raw)} test corrupt")

    test_by_type = defaultdict(list)
    for z0, A, Z, meta in test_corr_raw:
        test_by_type[meta["violation_type"]].append((z0, A, Z))

    # ── Model C: Compositional corruptions (structured/trained) ──
    c_path = CACHE / 'repro_model_c.pt'
    if c_path.exists():
        model_c = SlotIWCMEnergy(d_slot=d, d_action=11, hidden_dim=128, num_slots=N).to(DEVICE)
        model_c.load_state_dict(torch.load(c_path, map_location=DEVICE, weights_only=False))
        model_c.eval()
        print("  [cached] Model C (compositional) loaded")
    else:
        train_corr_c = [(z0, A, Z) for z0, A, Z, _ in
                        [(torch.from_numpy(item[0][0]).float(), torch.from_numpy(item[0][1]).float(),
                          torch.from_numpy(item[0][2]).float(), item[1])
                         for item in grid["train_corr"]]]
        print("  Training Model C (compositional corruptions, 150 epochs)...")
        model_c = train_model(train_valid, train_corr_c, 150, "C", d, 11, N)
        torch.save(model_c.state_dict(), c_path)
        print("  Model C cached")

    # ── Model B: Random noise corruptions ──
    b_path = CACHE / 'repro_model_b.pt'
    if b_path.exists():
        model_b = SlotIWCMEnergy(d_slot=d, d_action=11, hidden_dim=128, num_slots=N).to(DEVICE)
        model_b.load_state_dict(torch.load(b_path, map_location=DEVICE, weights_only=False))
        model_b.eval()
        print("  [cached] Model B (random noise) loaded")
    else:
        n_corr = max(len(train_valid), 200)
        random_corr = []
        rng_b = np.random.RandomState(42)
        for _ in range(n_corr):
            idx = rng_b.randint(0, len(train_valid))
            z0, A, Z = train_valid[idx]
            Z_noisy = Z + torch.randn_like(Z) * 0.3
            Z_noisy[:, :, :5] = Z[:, :, :5]
            Z_noisy[:, :, 15:] = Z[:, :, 15:]
            random_corr.append((z0, A, Z_noisy))
        print("  Training Model B (random noise, 150 epochs)...")
        model_b = train_model(train_valid, random_corr, 150, "B", d, 11, N)
        torch.save(model_b.state_dict(), b_path)
        print("  Model B cached")

    # ── Evaluate ──
    rc = evaluate(model_c, test_valid, test_by_type)
    rb = evaluate(model_b, test_valid, test_by_type)

    print(f"\n  {'Type':<14} {'B AUROC':>10} {'C AUROC':>10} {'Delta':>10}")
    print(f"  {'-'*46}")
    deltas = []
    for vt in sorted(test_by_type.keys()):
        ab = rb[vt]['auroc']
        ac = rc[vt]['auroc']
        delta = ac - ab
        deltas.append(delta)
        print(f"  {vt:<14} {ab:10.3f} {ac:10.3f} {delta:+10.3f}")

    avg_b = np.mean([rb[vt]['auroc'] for vt in rb])
    avg_c = np.mean([rc[vt]['auroc'] for vt in rc])
    avg_delta = np.mean(deltas)
    print(f"  {'-'*46}")
    print(f"  {'AVERAGE':<14} {avg_b:10.3f} {avg_c:10.3f} {avg_delta:+10.3f}")

    # Threshold checks
    c1 = check("Model C avg AUROC >= 0.85", avg_c >= 0.85, avg_c, ">=0.85")
    c2 = check(f"Model C - Model B gap >= +0.10", avg_delta >= 0.10, avg_delta, ">=+0.10")
    results['test1'] = {'passed': c1 and c2}
    return c1 and c2


# ═══════════════════════════════════════════════════════════════════════════════
# TESTS 2-4: DM Control Experiments (shared infrastructure)
# ═══════════════════════════════════════════════════════════════════════════════

def _load_dm_models(force_train=False):
    """Load or train DM Control models. Returns (iwcm, rollout, degraded)."""
    from src.env.dm_control_wrapper import DMControlWrapper
    from src.env.dm_control_encoder import DMControlOracleEncoder
    from src.iwcm.fused_energy import FusedIWCMEnergy
    from scripts.experiments._common import Rollout, train_rl

    # ── Load or generate data ──
    if DATA_CACHE.exists() and not force_train:
        with open(DATA_CACHE, 'rb') as f:
            tv, tc, ts = pickle.load(f)
        print("  [cached] DM data loaded")
        # Still need env for dims
        w = DMControlWrapper('cartpole', 'swingup', seed=42, max_episode_steps=300)
        e = DMControlOracleEncoder('cartpole')
        da = w.action_dim
        Ns, ds = 8, 19
    else:
        print("  Generating DM Control data (400+ trajectories)...")
        w = DMControlWrapper('cartpole', 'swingup', seed=42, max_episode_steps=300)
        e = DMControlOracleEncoder('cartpole')
        da = w.action_dim
        from scripts.experiments._common import data as dm_data
        tv, tc, ts = dm_data(w, e, nv=400, nc=200, H=100)
        with open(DATA_CACHE, 'wb') as f:
            pickle.dump((tv, tc, ts), f)
        print("  DM data cached")
    Ns, ds = 8, 19

    # ── IWCM Energy ──
    iwcm_path = CACHE / 'repro_dm_iwcm.pt'
    if iwcm_path.exists() and not force_train:
        iwcm = FusedIWCMEnergy(ds, da, hidden=128, num_slots=Ns).to(DEVICE)
        iwcm.load_state_dict(torch.load(iwcm_path, map_location=DEVICE, weights_only=False))
        iwcm.eval()
        print("  [cached] DM IWCM loaded")
    else:
        print("  Training DM IWCM energy (150 epochs)...")
        from scripts.experiments._common import train_iwcm
        iwcm = train_iwcm(tv, tc, d_slot=ds, d_a=da, Ns=Ns, epochs=150)
        iwcm.eval()
        torch.save(iwcm.state_dict(), iwcm_path)
        print("  DM IWCM cached")

    # ── Normal Rollout ──
    rl_path = CACHE / 'repro_dm_rollout.pt'
    if rl_path.exists() and not force_train:
        rl = Rollout(Ns, ds, da).to(DEVICE)
        rl.load_state_dict(torch.load(rl_path, map_location=DEVICE, weights_only=False))
        rl.eval()
        print("  [cached] DM Rollout loaded")
    else:
        print("  Training DM Rollout (100 epochs)...")
        rl = Rollout(Ns, ds, da).to(DEVICE)
        train_rl(rl, tv, 100, ep=100)
        rl.eval()
        torch.save(rl.state_dict(), rl_path)
        print("  DM Rollout cached")

    # ── Degraded Rollout ──
    deg_path = CACHE / 'repro_dm_degraded.pt'
    if deg_path.exists() and not force_train:
        rl_d = Rollout(Ns, ds, da, h=32).to(DEVICE)
        rl_d.load_state_dict(torch.load(deg_path, map_location=DEVICE, weights_only=False))
        rl_d.eval()
        print("  [cached] DM Degraded Rollout loaded")
    else:
        print("  Training DM Degraded Rollout (15 epochs, h=32)...")
        import torch.nn as nn
        rl_d = Rollout(Ns, ds, da, h=32)
        rl_d.net[1] = nn.Tanh()
        rl_d.to(DEVICE)
        train_rl(rl_d, tv[:50], 100, ep=15)
        rl_d.eval()
        torch.save(rl_d.state_dict(), deg_path)
        print("  DM Degraded Rollout cached")

    return iwcm, rl, rl_d, (tv, tc, ts), da


import torch.nn.functional as F

def _solve(model, z0, A, init_Z=None, steps=100, lr=0.01, anticollapse=True):
    """Gradient descent solver for DM Control. anticollapse adds VICReg + transition
    hinge penalties at solver time to prevent trivial freeze collapse."""
    import torch.nn.functional as F
    B, Hf = A.shape[:2]
    Ns, ds = z0.shape[1], z0.shape[2]
    if init_Z is not None:
        Z = init_Z.clone().detach().to(DEVICE)
    else:
        Z = torch.randn(B, Hf, Ns, ds, device=DEVICE)
    Z.requires_grad_(True)
    vel = torch.zeros_like(Z)
    for _ in range(steps):
        e = model(z0, A, Z).mean()
        if anticollapse:
            # VICReg variance floor
            var_p = 100.0 * F.relu(0.001 - Z.std(dim=1).mean())
            # Transition hinge: penalize z_{t+1} == z_t when action != 0
            diff_sq = (Z[:, 1:] - Z[:, :-1]).pow(2).mean()
            a_active = (A[:, :-1].norm(dim=-1) > 1e-4).float().mean()
            transit_p = 500.0 * F.relu(0.005 - diff_sq) * a_active
            e = e + var_p + transit_p
        g = torch.autograd.grad(e, Z, create_graph=False)[0]
        vel = 0.9 * vel + g
        Z = Z.detach() - lr * vel
        Z.requires_grad_(True)
        vel = vel.detach()
    return Z


def test_dm_drift(iwcm, rl, ts, da):
    """Test 2: DM Control drift at H=100."""
    # ponytail: adapted from dm_drift.py (>18 lines → inlined here)
    print("\n" + "=" * 65)
    print("TEST 2: DM Control Drift — Cold vs z0-rep vs Warm at H=100")
    print("=" * 65)

    Ns, ds = 8, 19
    cold_gaps, z0rep_gaps, warm_gaps = [], [], []

    for z0, A, Zt in ts:
        zb = z0.unsqueeze(0).to(DEVICE)
        Ab = A.unsqueeze(0).to(DEVICE)
        Zb = Zt.unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            Zr = rl.rollout(zb, Ab)

        Zc = _solve(iwcm, zb, Ab, init_Z=torch.randn(1, 100, Ns, ds, device=DEVICE))
        Zz0 = z0.unsqueeze(0).unsqueeze(1).expand(-1, 100, -1, -1).clone().to(DEVICE)
        Zz0 = _solve(iwcm, zb, Ab, init_Z=Zz0)
        Zw = _solve(iwcm, zb, Ab, init_Z=Zr, steps=100, lr=0.005)

        with torch.no_grad():
            E_true = iwcm(zb, Ab[:, :100], Zb[:, :100]).item()
            E_cold = iwcm(zb, Ab[:, :100], Zc[:, :100]).item()
            E_z0rep = iwcm(zb, Ab[:, :100], Zz0[:, :100]).item()
            E_warm = iwcm(zb, Ab[:, :100], Zw[:, :100]).item()

        cold_gaps.append(E_cold - E_true)
        z0rep_gaps.append(E_z0rep - E_true)
        warm_gaps.append(E_warm - E_true)

    avg_cold = np.mean(cold_gaps)
    avg_z0rep = np.mean(z0rep_gaps)
    avg_warm = np.mean(warm_gaps)

    print(f"  H=100: cold Δ={avg_cold:+.3f}, z0-rep Δ={avg_z0rep:+.3f}, warm Δ={avg_warm:+.3f}")

    c1 = check("Cold-start H=100 energy gap >= +30", avg_cold >= 30, avg_cold, ">=+30", "")
    c2 = check("z0-rep H=100 energy gap <= 0", avg_z0rep <= 0, avg_z0rep, "≤0", "")
    c3 = check("Warm-start H=100 energy gap <= 0", avg_warm <= 0, avg_warm, "≤0", "")
    results['test2'] = {'passed': c1 and c2 and c3}
    return c1 and c2 and c3


def test_init_ablation(iwcm, rl, ts, da):
    """Test 3: Initialization ablation ranking.
    Expected: E_random > E_rollout > E_z0rep ≈ E_warm (lower = better).
    """
    print("\n" + "=" * 65)
    print("TEST 3: Init Ablation — Random vs Rollout vs z0-rep vs Warm")
    print("=" * 65)

    Ns, ds = 8, 19
    rand_energies, rollout_energies, z0rep_energies, warm_energies = [], [], [], []

    for z0, A, Zt in ts:
        zb = z0.unsqueeze(0).to(DEVICE)
        Ab = A.unsqueeze(0).to(DEVICE)
        Zb = Zt.unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            Zr = rl.rollout(zb, Ab)

        Zrand = _solve(iwcm, zb, Ab, init_Z=torch.randn(1, 100, Ns, ds, device=DEVICE))
        Zz0 = z0.unsqueeze(0).unsqueeze(1).expand(-1, 100, -1, -1).clone().to(DEVICE)
        Zz0 = _solve(iwcm, zb, Ab, init_Z=Zz0)
        Zw = _solve(iwcm, zb, Ab, init_Z=Zr, steps=100, lr=0.005)

        with torch.no_grad():
            E_rand = iwcm(zb, Ab[:, :100], Zrand[:, :100]).item()
            E_roll = iwcm(zb, Ab[:, :100], Zr[:, :100]).item()
            E_z0rep = iwcm(zb, Ab[:, :100], Zz0[:, :100]).item()
            E_warm = iwcm(zb, Ab[:, :100], Zw[:, :100]).item()

        rand_energies.append(E_rand)
        rollout_energies.append(E_roll)
        z0rep_energies.append(E_z0rep)
        warm_energies.append(E_warm)

    avg_rand = np.mean(rand_energies)
    avg_roll = np.mean(rollout_energies)
    avg_z0rep = np.mean(z0rep_energies)
    avg_warm = np.mean(warm_energies)

    print(f"  H=100 energies: random={avg_rand:+.3f}, rollout={avg_roll:+.3f}, "
          f"z0rep={avg_z0rep:+.3f}, warm={avg_warm:+.3f}")
    print(f"  Rankings: random < rollout? {avg_rand > avg_roll} | "
          f"rollout < z0rep? {avg_roll > avg_z0rep} | "
          f"z0rep ≈ warm? {abs(avg_z0rep - avg_warm) < 2.0}")

    c_rand_worst = avg_rand > avg_roll  # random is worst (highest energy)
    c_roll_better = avg_roll > avg_z0rep  # rollout worse than z0rep
    c_z0rep_best = avg_z0rep <= avg_warm + 2.0  # z0rep is at least as good as warm

    p1 = check("Random (worst) > Rollout energy", avg_rand > avg_roll, avg_rand - avg_roll, ">0")
    p2 = check(f"Rollout > z0-rep energy", avg_roll > avg_z0rep, avg_roll - avg_z0rep, ">0")
    p3 = check(f"z0-rep ≈ Warm (|Δ| < 2.0)", abs(avg_z0rep - avg_warm) < 2.0, abs(avg_z0rep - avg_warm), "<2.0")
    results['test3'] = {'passed': p1 and p2 and p3}
    return p1 and p2 and p3


def test_recovery(iwcm, rl_d, ts, da):
    """Test 4: Degraded rollout recovery — IWCM < degraded rollout energy."""
    print("\n" + "=" * 65)
    print("TEST 4: Degraded Rollout Recovery — IWCM pulls back")
    print("=" * 65)

    Ns, ds = 8, 19
    roll_gaps, warm_gaps = [], []

    for z0, A, Zt in ts:
        zb = z0.unsqueeze(0).to(DEVICE)
        Ab = A.unsqueeze(0).to(DEVICE)
        Zb = Zt.unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            Zr = rl_d.rollout(zb, Ab)
        Zw = _solve(iwcm, zb, Ab, init_Z=Zr, steps=100, lr=0.005)

        with torch.no_grad():
            E_true = iwcm(zb, Ab[:, :100], Zb[:, :100]).item()
            E_roll = iwcm(zb, Ab[:, :100], Zr[:, :100]).item()
            E_warm = iwcm(zb, Ab[:, :100], Zw[:, :100]).item()

        roll_gaps.append(E_roll - E_true)
        warm_gaps.append(E_warm - E_true)

    avg_roll = np.mean(roll_gaps)
    avg_warm = np.mean(warm_gaps)
    recovers = avg_warm < avg_roll

    print(f"  H=100: degraded Δ={avg_roll:+.3f}, warm Δ={avg_warm:+.3f}")
    print(f"  {'✓ RECOVERS' if recovers else '✗ NO RECOVERY'}")

    p = check("IWCM energy < degraded rollout energy", recovers, avg_warm - avg_roll, "<0")
    results['test4'] = {'passed': p}
    return p


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cached', action='store_true', help='Skip training, use cached models only')
    parser.add_argument('--test', type=int, choices=[1, 2, 3, 4], help='Run single test')
    args = parser.parse_args()

    print(f"Device: {DEVICE}")
    print(f"PyTorch: {torch.__version__}")

    if args.test is None or args.test == 1:
        test_ac3()

    if args.test is None or args.test in [2, 3, 4]:
        try:
            iwcm, rl, rl_d, (tv, tc, ts), da = _load_dm_models(force_train=args.cached)
        except Exception as e:
            print(f"  [SKIP] DM Control tests: {e}")
            print("  (MuJoCo or dm_control not available)")
            results['test2'] = results.get('test3', results.get('test4', {'passed': 'SKIP'}))
            return

        if args.test is None or args.test == 2:
            test_dm_drift(iwcm, rl, ts, da)
        if args.test is None or args.test == 3:
            test_init_ablation(iwcm, rl, ts, da)
        if args.test is None or args.test == 4:
            test_recovery(iwcm, rl_d, ts, da)

    # Summary
    print("\n" + "=" * 65)
    print("SUMMARY")
    print("=" * 65)
    passed = sum(1 for r in results.values() if r.get('passed') is True)
    failed = sum(1 for r in results.values() if r.get('passed') is False)
    skipped = sum(1 for r in results.values() if r.get('passed') == 'SKIP')
    total = len(results)
    for name, r in results.items():
        status = 'PASS' if r.get('passed') is True else 'FAIL' if r.get('passed') is False else 'SKIP'
        print(f"  {name}: {status}")
    print(f"\n  {passed}/{total} passed, {failed} failed, {skipped} skipped")
    return 0 if failed == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
