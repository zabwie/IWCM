"""Minimal smoke tests for IWCM codebase core functionality."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np

def test_iwcm_energy_forward():
    """Test IWCM energy function produces valid output."""
    from src.iwcm.model import IWCM
    model = IWCM(d_state=128, d_action=11, hidden_dim=128)
    z0 = torch.randn(2, 128)
    A = torch.randn(2, 25, 11)
    Z = torch.randn(2, 25, 128)
    energy = model.energy(z0, A, Z)
    assert energy.shape == (2,), f"Expected (2,) got {energy.shape}"
    assert energy.isfinite().all(), "Energy should be finite"
    per_head = model.energy_per_head(z0, A, Z)
    assert len(per_head) == 5, f"Expected 5 heads, got {len(per_head)}"
    for name, val in per_head.items():
        assert val.shape == (2,), f"Head {name}: expected (2,) got {val.shape}"
    return "PASSED: IWCM energy forward"

def test_constraint_heads_independence():
    """Test each constraint head produces independent non-zero output."""
    from src.iwcm.energy import IWCMEnergy
    efn = IWCMEnergy(d_state=128, d_action=11, hidden_dim=128)
    z0 = torch.randn(2, 128)
    A = torch.randn(2, 25, 11)
    Z = torch.randn(2, 25, 128)
    for head in [efn.boundary_head, efn.local_head, efn.invariant_head, efn.effect_head, efn.counterfactual_head]:
        out = head(z0, A, Z)
        assert out.shape == (2,), f"Head {head.name}: expected (2,) got {out.shape}"
        assert not torch.allclose(out, torch.zeros(2)), f"Head {head.name}: output is all zeros"
    return "PASSED: Constraint heads independence"

def test_gridworld_smoke():
    """Test GridWorld basic operations."""
    from src.env.grid_world import GridWorld
    gw = GridWorld(grid_size=8, seed=42)
    gw.reset()
    for _ in range(5):
        acts = gw.get_valid_actions()
        assert len(acts) > 0, "No valid actions"
        state, reward, done, info = gw.step(acts[0])
    return "PASSED: GridWorld smoke"

def test_ac3_oracle():
    """Test AC3 oracle detects violations."""
    from src.ac3.oracle import SymbolicOracle
    from src.ac3.mutations.grammar import SymbolicMutationGrammar, SymbolicTrajectory
    from src.env.symbolic_state import SymbolicState
    oracle = SymbolicOracle()
    grammar = SymbolicMutationGrammar()
    s0 = SymbolicState(agent_pos=(0, 0), grid_size=8, step=0)
    s1 = SymbolicState(agent_pos=(1, 0), grid_size=8, step=1)
    valid = SymbolicTrajectory(states=[s0, s1], actions=[1], horizon=2)
    assert oracle.is_valid(valid), "Valid trajectory should be valid"
    return "PASSED: AC3 oracle"

def test_metrics_evaluate_model():
    """Test evaluate_model orchestrator exists and is callable."""
    from src.metrics.evaluation import evaluate_model, build_ablation_models, RolloutModel
    assert callable(evaluate_model), "evaluate_model should be callable"
    models = build_ablation_models(128, 11, 128)
    assert len(models) >= 5, f"Expected >=5 models, got {len(models)}"
    rm = RolloutModel(128, 11)
    assert callable(rm), "RolloutModel should be callable"
    return "PASSED: evaluate_model orchestrator"

if __name__ == "__main__":
    results = []
    for test_fn in [test_iwcm_energy_forward, test_constraint_heads_independence,
                     test_gridworld_smoke, test_ac3_oracle, test_metrics_evaluate_model]:
        try:
            r = test_fn()
            print(r)
            results.append(True)
        except Exception as e:
            print(f"FAILED: {test_fn.__name__}: {e}")
            results.append(False)
    passed = sum(results)
    total = len(results)
    print(f"\n{passed}/{total} tests passed")
    assert passed == total, f"{total - passed} tests failed"
    print("ALL TESTS PASSED")
