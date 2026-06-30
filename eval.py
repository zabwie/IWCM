#!/usr/bin/env python3
"""One-command evaluation: reproduction tests + rigorous proofs + visual proof.

Usage:
    python eval.py                     # cached models (30 sec) — default
    python eval.py --fresh             # retrain everything (~20 min)
    python eval.py --skip-visual       # skip video generation

Outputs:
  - Terminal: reproduction PASS/FAIL, statistical tests, validity analysis
  - outputs/proof/*.png + *.mp4       — all proof figures
  - outputs/checkpoints/repro_*.pt    — cached model files
"""
import sys, subprocess, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
START = time.time()

def sec(t): return f"{t:.0f}s" if t < 120 else f"{t/60:.1f}m"

def run(cmd, label):
    print(f"\n{'='*65}")
    print(f"  {label}")
    print(f"{'='*65}\n")
    t0 = time.time()
    r = subprocess.run(cmd, cwd=ROOT)
    dt = time.time() - t0
    print(f"\n  [{label}] finished in {sec(dt)}, exit={r.returncode}")
    if r.returncode != 0:
        print(f"  WARNING: non-zero exit code {r.returncode}")
    return r.returncode

if __name__ == '__main__':
    fresh = '--fresh' in sys.argv
    skip_visual = '--skip-visual' in sys.argv

    # Step 1: reproduction tests (cached by default)
    repro_args = [sys.executable, 'scripts/reproduction.py']
    if not fresh:
        repro_args.append('--cached')
    run(repro_args, "Reproduction Tests (PASS/FAIL)")

    # Step 2: rigorous proofs (statistics, validity, figures)
    run([sys.executable, 'scripts/proof_rigorous.py'], "Rigorous Proofs (6 proofs)")

    # Step 3: visual proof (video + energy drift + grid-world)
    if not skip_visual:
        run([sys.executable, 'scripts/proof_visual.py'], "Visual Proof (video + plots)")

    total = time.time() - START
    print(f"\n{'='*65}")
    print(f"  ALL DONE in {sec(total)}")
    print(f"  Outputs in: outputs/proof/")
    print(f"  Proof summary: REPROOF.md")
    print(f"{'='*65}")
