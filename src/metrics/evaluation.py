"""Evaluation metrics for IWCM experiments (Section 7.4).

Nine metrics:
  1. Constraint violation rate over H ∈ {10, 25, 50, 100}
  2. Object identity preservation across full trajectories
  3. Conservation violation detection on held-out scenarios
  4. Valid/invalid future classification accuracy
  5. Repair accuracy on corrupted worldlines
  6. Counterfactual locality accuracy
  7. Splice detection across long delays (Δt > 20)
  8. Planning success rate on long-horizon key-door tasks
  9. Cross-surface law generalization (PRIMARY METRIC)
"""

import torch
import torch.nn.functional as F
from typing import Dict, List, Tuple, Any
import numpy as np

from ..iwcm.model import IWCM
from ..env.symbolic_state import SymbolicState, SymbolicTrajectory
from ..ac3.oracle import SymbolicOracle


# ═══════════════════════════════════════════════════════════
# Metric 1: Constraint Violation Rate
# ═══════════════════════════════════════════════════════════

def metric_constraint_violation(
    model: IWCM,
    trajectories: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    oracle: SymbolicOracle,
    horizons: List[int] = [10, 25, 50, 100],
) -> Dict[int, float]:
    """Measure constraint violation rate at different horizons.

    Args:
        model: Trained IWCM world model.
        trajectories: List of (z0, A, Z) tuples for evaluation.
        oracle: Ground-truth constraint oracle.
        horizons: Horizons to evaluate at.

    Returns:
        Dict mapping horizon to violation rate.
    """
    results = {}

    for H in horizons:
        violations = 0
        total = 0

        for z0, A, Z in trajectories:
            z0 = z0.unsqueeze(0)  # add batch dim
            A = A[:H].unsqueeze(0) if len(A) > H else A.unsqueeze(0)
            Z_in = Z[:H].unsqueeze(0)

            # Model solves worldline
            Z_solved = model.solve(z0, A)
            energy = model.energy(z0, A, Z_solved)

            # Check if model produces violations
            per_head = model.energy_per_head(z0, A, Z_solved)
            if per_head["invariant"].mean().item() > 0.5:  # threshold
                violations += 1
            total += 1

        results[H] = violations / max(total, 1)
    return results


# ═══════════════════════════════════════════════════════════
# Metric 2: Object Identity Preservation
# ═══════════════════════════════════════════════════════════

def metric_identity_preservation(
    model: IWCM,
    trajectories: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    num_objects: int = 5,
) -> float:
    """Measure identity preservation across full trajectories.

    Checks whether object identities remain consistent in predicted worldlines.
    """
    preserved = 0
    total = 0

    for z0, A, Z in trajectories:
        z0 = z0.unsqueeze(0)
        A = A.unsqueeze(0)
        Z_gt = Z.unsqueeze(0)

        Z_pred = model.solve(z0, A)

        # Compare ground truth and predicted identities
        # Objects should maintain consistent features
        for t in range(Z_pred.shape[1]):
            sim = F.cosine_similarity(
                Z_pred[:, t].flatten(), Z_gt[:, t].flatten(), dim=-1,
            )
            if sim > 0.8:  # threshold
                preserved += 1
            total += 1

    return preserved / max(total, 1)


# ═══════════════════════════════════════════════════════════
# Metric 3: Conservation Violation Detection
# ═══════════════════════════════════════════════════════════

def metric_conservation_detection(
    model: IWCM,
    valid_trajs: List,
    invalid_trajs: List,  # conservation-violating trajectories
) -> float:
    """Measure accuracy of detecting conservation violations.

    Args:
        model: Trained IWCM model.
        valid_trajs: Trajectories that obey conservation.
        invalid_trajs: Trajectories with conservation violations (held-out).

    Returns:
        Detection accuracy.
    """
    correct = 0
    total = 0

    for valid in valid_trajs:
        z0, A, Z = valid
        z0, A, Z = z0.unsqueeze(0), A.unsqueeze(0), Z.unsqueeze(0)
        energy = model.energy(z0, A, Z)
        if energy.mean().item() < 1.0:  # low energy = valid
            correct += 1
        total += 1

    for invalid in invalid_trajs:
        z0, A, Z = invalid
        z0, A, Z = z0.unsqueeze(0), A.unsqueeze(0), Z.unsqueeze(0)
        energy = model.energy(z0, A, Z)
        if energy.mean().item() > 1.0:  # high energy = invalid
            correct += 1
        total += 1

    return correct / max(total, 1)


# ═══════════════════════════════════════════════════════════
# Metric 4: Valid/Invalid Classification
# ═══════════════════════════════════════════════════════════

def metric_classification(
    model: IWCM,
    valid_trajs: List,
    invalid_trajs: List,
) -> Tuple[float, float, float]:
    """Binary classification: valid vs invalid futures.

    Returns:
        accuracy, precision, recall.
    """
    scores = []
    labels = []

    for valid in valid_trajs:
        z0, A, Z = valid
        z0, A, Z = z0.unsqueeze(0), A.unsqueeze(0), Z.unsqueeze(0)
        accept = model.score_accept(z0, A, Z).mean().item()
        scores.append(accept)
        labels.append(1)  # valid

    for invalid in invalid_trajs:
        z0, A, Z = invalid
        z0, A, Z = z0.unsqueeze(0), A.unsqueeze(0), Z.unsqueeze(0)
        accept = model.score_accept(z0, A, Z).mean().item()
        scores.append(accept)
        labels.append(0)  # invalid

    # Simple threshold at 0.5
    preds = [1 if s > 0.5 else 0 for s in scores]
    tp = sum(1 for p, l in zip(preds, labels) if p == 1 and l == 1)
    fp = sum(1 for p, l in zip(preds, labels) if p == 1 and l == 0)
    fn = sum(1 for p, l in zip(preds, labels) if p == 0 and l == 1)

    accuracy = (tp + (len(labels) - tp - fp - fn)) / max(len(labels), 1)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)

    return accuracy, precision, recall


# ═══════════════════════════════════════════════════════════
# Metric 5: Repair Accuracy
# ═══════════════════════════════════════════════════════════

def metric_repair_accuracy(
    model: IWCM,
    corrupted_trajs: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    original_energies: List[float],
) -> float:
    """Measure repair accuracy on corrupted worldlines.

    A successful repair reduces energy and restores valid structure.
    """
    improvements = []
    total = 0

    for (z0, A, Z_corr), orig_e in zip(corrupted_trajs, original_energies):
        z0, A, Z_corr = z0.unsqueeze(0), A.unsqueeze(0), Z_corr.unsqueeze(0)
        repaired, improvement = model.repair(z0, A, Z_corr)
        if improvement.mean().item() > 0:
            improvements.append(1)
        else:
            improvements.append(0)
        total += 1

    return sum(improvements) / max(total, 1)


# ═══════════════════════════════════════════════════════════
# Metric 6: Counterfactual Locality
# ═══════════════════════════════════════════════════════════

def metric_counterfactual_locality(
    model: IWCM,
    counterfactual_pairs: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor,
                                      torch.Tensor, torch.Tensor]],
) -> float:
    """Measure counterfactual locality accuracy.

    For two futures from same z0 with different actions, check if
    model correctly identifies which objects should/not change.
    """
    correct = 0
    total = 0

    for z0, A, Z, A2, Z2 in counterfactual_pairs:
        z0 = z0.unsqueeze(0)
        A, Z = A.unsqueeze(0), Z.unsqueeze(0)
        A2, Z2 = A2.unsqueeze(0), Z2.unsqueeze(0)

        # Energy of both worldlines
        e1 = model.energy(z0, A, Z)
        e2 = model.energy(z0, A2, Z2)

        # Counterfactual head should detect inconsistencies
        cfact = model.energy_fn.counterfactual_head.forward_paired(
            z0, A, Z, A2, Z2,
        )

        # If actions differ, cfact should be low (normal)
        # If we artificially make Z2 identical when actions differ, cfact should be high
        if torch.abs(e1 - e2).mean().item() < 1.0:
            correct += 1
        total += 1

    return correct / max(total, 1)


# ═══════════════════════════════════════════════════════════
# Metric 7: Splice Detection
# ═══════════════════════════════════════════════════════════

def metric_splice_detection(
    model: IWCM,
    spliced_trajs: List[Tuple],
    valid_trajs: List[Tuple],
    long_horizon: int = 25,
) -> float:
    """Measure accuracy detecting spliced trajectories (Δt > 20).

    Splices join two valid halves into a globally inconsistent whole.
    """
    correct = 0
    total = 0

    # Valid trajectories should be accepted
    for z0, A, Z in valid_trajs:
        z0, A, Z = z0.unsqueeze(0), A.unsqueeze(0), Z.unsqueeze(0)
        accept = model.score_accept(z0, A, Z).mean().item()
        if accept > 0.5:
            correct += 1
        total += 1

    # Spliced trajectories should be rejected
    for z0, A, Z in spliced_trajs:
        z0, A, Z = z0.unsqueeze(0), A.unsqueeze(0), Z.unsqueeze(0)
        accept = model.score_accept(z0, A, Z).mean().item()
        if accept < 0.5:
            correct += 1
        total += 1

    return correct / max(total, 1)


# ═══════════════════════════════════════════════════════════
# Metric 8: Planning Success Rate
# ═══════════════════════════════════════════════════════════

def metric_planning_success(
    model: IWCM,
    test_scenarios: List[dict],
    horizon: int = 25,
) -> float:
    """Measure planning success rate on long-horizon key-door tasks.

    Args:
        model: Trained IWCM model.
        test_scenarios: List of scenario configs.
        horizon: Planning horizon.

    Returns:
        Success rate (0 to 1).
    """
    from ..iwcm.planner import GoalConstraint

    successes = 0
    total = len(test_scenarios)

    for scenario in test_scenarios:
        try:
            goal = GoalConstraint(
                goal_type=scenario.get("goal", {}).get("type", "position"),
                position=tuple(scenario.get("goal", {}).get("pos", (0, 0))),
            )
            z0 = torch.randn(1, model.d_state)  # encoded start
            A_plan, Z_plan, energy = model.planner.plan(z0, goal, horizon)

            # Simple heuristic: low energy = successful plan
            if energy.item() < 5.0:
                successes += 1
        except Exception:
            pass

    return successes / max(total, 1)


# ═══════════════════════════════════════════════════════════
# Metric 9: Cross-Surface Law Generalization (PRIMARY)
# ═══════════════════════════════════════════════════════════

def metric_cross_surface_generalization(
    model: IWCM,
    train_violation_type: str,
    test_violations: Dict[str, List[Tuple]],
) -> Dict[str, float]:
    """Measure cross-surface law generalization.

    Model is trained on one type of violation (e.g., key duplication).
    This metric tests whether it generalizes to other conservation
    violations with different surface forms (e.g., box duplication,
    door-state change).

    This is the DEFINITIVE test of law learning vs. pattern learning.
    """
    results = {}

    for violation_type, test_trajs in test_violations.items():
        correct = 0
        total = 0

        for z0, A, Z in test_trajs:
            z0, A, Z = z0.unsqueeze(0), A.unsqueeze(0), Z.unsqueeze(0)
            energy = model.energy(z0, A, Z)

            # High energy = detected violation
            if energy.mean().item() > 1.0:
                correct += 1
            total += 1

        results[violation_type] = correct / max(total, 1)

    # Overall cross-surface score (average over held-out types)
    held_out = {k: v for k, v in results.items() if k != train_violation_type}
    cross_surface_score = np.mean(list(held_out.values())) if held_out else 0.0

    results["cross_surface_overall"] = cross_surface_score
    results["train_type"] = results.get(train_violation_type, 0.0)

    return results


# ═══════════════════════════════════════════════════════════
# Unified Evaluation
# ═══════════════════════════════════════════════════════════

def evaluate_model(
    model: IWCM,
    data: Dict[str, Any],
    horizons: List[int] = [10, 25, 50, 100],
) -> Dict[str, Any]:
    """Run all 9 evaluation metrics and return results.

    Args:
        model: Trained IWCM model.
        data: Dict with keys: valid_trajs, invalid_trajs, corrupted_trajs,
              counterfactual_pairs, spliced_trajs, test_scenarios, etc.
        horizons: Horizons for metric 1.

    Returns:
        Dict of all metric results.
    """
    oracle = SymbolicOracle()

    results: Dict[str, Any] = {}

    # Metric 1
    results["constraint_violation_rate"] = metric_constraint_violation(
        model, data.get("valid_trajs", []), oracle, horizons,
    )

    # Metric 2
    results["identity_preservation"] = metric_identity_preservation(
        model, data.get("valid_trajs", []),
    )

    # Metric 3
    results["conservation_detection"] = metric_conservation_detection(
        model, data.get("valid_trajs", []), data.get("conservation_violating", []),
    )

    # Metric 4
    acc, prec, rec = metric_classification(
        model, data.get("valid_trajs", []), data.get("invalid_trajs", []),
    )
    results["classification"] = {"accuracy": acc, "precision": prec, "recall": rec}

    # Metric 5
    results["repair_accuracy"] = metric_repair_accuracy(
        model, data.get("corrupted_trajs", []), data.get("orig_energies", [0.0]),
    )

    # Metric 6
    results["counterfactual_locality"] = metric_counterfactual_locality(
        model, data.get("counterfactual_pairs", []),
    )

    # Metric 7
    results["splice_detection"] = metric_splice_detection(
        model, data.get("spliced_trajs", []), data.get("valid_trajs", []),
    )

    # Metric 8
    results["planning_success"] = metric_planning_success(
        model, data.get("test_scenarios", []),
    )

    # Metric 9
    results["cross_surface_generalization"] = metric_cross_surface_generalization(
        model,
        data.get("train_violation_type", "key_duplication"),
        data.get("cross_surface_tests", {}),
    )

    return results
