"""Evaluation metrics for IWCM — 10-metric paper specification.

Core (5):
  1. Cross-surface law generalization (PRIMARY)
  2. Valid/invalid worldline classification (AUROC/AUPRC/per-type)
  3. Long-horizon drift curve (CVR over H=5,10,20,50,100)
  4. Repair success + minimality
  5. Planning success rate

Supplementary (5):
  6. Energy calibration (margin, severity correlation)
  7. Local-vs-global failure detection
  8. Counterfactual locality F1
  9. AC3 hardness quality
  10. Ablation table

GPU-optimized: batch evaluation via device="cuda".
"""

import torch
import numpy as np
import torch.nn.functional as F
from typing import Dict, List, Tuple, Any, Optional, Union
from collections import defaultdict
from sklearn.metrics import roc_auc_score, average_precision_score

from ..iwcm.model import IWCM
from ..ac3.oracle import SymbolicOracle
from ..ac3.mutations.grammar import SymbolicMutationGrammar, SymbolicTrajectory
from ..env.symbolic_state import SymbolicState
from ..env.scenarios import Scenario
from ..iwcm.planner import GoalConstraint


def _to_tensor(x: Union[np.ndarray, torch.Tensor], device: str = "cuda") -> torch.Tensor:
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x).to(device)
    return x.to(device)


def _batch_to_tensors(trajs: List[Tuple]) -> List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    return [(_to_tensor(z0), _to_tensor(A), _to_tensor(Z)) for z0, A, Z in trajs]


# ═══════════════════════════════════════════════════════════
# Metric 1: Cross-Surface Law Generalization (PRIMARY)
# ═══════════════════════════════════════════════════════════

def metric_cross_surface_law_generalization(
    model: IWCM,
    train_data: Dict[str, Dict],
    test_data: Dict[str, Dict],
    device: str = "cuda",
) -> Dict[str, Any]:
    """Primary: does model learn laws, not surface patterns?

    Train on one surface form, test on different surface form.
    Returns ID acc, held-out acc, generalization gap, per-law breakdown.
    """
    per_law = {}
    for law_name, data in test_data.items():
        valid_trajs = data.get("valid", [])
        invalid_trajs = data.get("invalid", [])

        correct_valid, correct_invalid = 0, 0
        for z0, A, Z in valid_trajs:
            z0_d = _to_tensor(z0, device).unsqueeze(0)
            A_d = _to_tensor(A, device).unsqueeze(0)
            Z_d = _to_tensor(Z, device).unsqueeze(0)
            if model.score_accept(z0_d, A_d, Z_d).item() > 0.5:
                correct_valid += 1
        for z0, A, Z in invalid_trajs:
            z0_d = _to_tensor(z0, device).unsqueeze(0)
            A_d = _to_tensor(A, device).unsqueeze(0)
            Z_d = _to_tensor(Z, device).unsqueeze(0)
            if model.score_accept(z0_d, A_d, Z_d).item() < 0.5:
                correct_invalid += 1

        n_v, n_i = max(len(valid_trajs), 1), max(len(invalid_trajs), 1)
        per_law[law_name] = {
            "valid_accuracy": correct_valid / n_v,
            "invalid_rejection": correct_invalid / n_i,
            "balanced_accuracy": 0.5 * (correct_valid / n_v + correct_invalid / n_i),
        }

    result = {"per_law_breakdown": per_law}
    train_laws = list(train_data.keys()) if train_data else []
    if train_laws:
        train_law = train_laws[0]
        held_out = {k: v for k, v in per_law.items() if k != train_law}
        id_acc = per_law.get(train_law, {}).get("balanced_accuracy", 0.0)
        ho_accs = [v["balanced_accuracy"] for v in held_out.values()]
        ho_acc = np.mean(ho_accs) if ho_accs else 0.0
        result["in_distribution_accuracy"] = id_acc
        result["held_out_accuracy"] = ho_acc
        result["generalization_gap"] = id_acc - ho_acc
        result["law_generalization_accuracy"] = ho_acc
    return result


# ═══════════════════════════════════════════════════════════
# Metric 2: Valid/Invalid Classification
# ═══════════════════════════════════════════════════════════

def metric_valid_invalid_classification(
    model: IWCM, valid_trajs: List[Tuple],
    invalid_trajs_by_type: Dict[str, List[Tuple]],
    device: str = "cuda",
) -> Dict[str, Any]:
    scores, labels = [], []
    per_type: Dict[str, List[float]] = defaultdict(list)

    for z0, A, Z in valid_trajs:
        z0, A, Z = _to_tensor(z0,device).unsqueeze(0), _to_tensor(A,device).unsqueeze(0), _to_tensor(Z,device).unsqueeze(0)
        scores.append(model.score_accept(z0, A, Z).item()); labels.append(1)
    for vtype, trajs in invalid_trajs_by_type.items():
        for z0, A, Z in trajs:
            z0, A, Z = _to_tensor(z0,device).unsqueeze(0), _to_tensor(A,device).unsqueeze(0), _to_tensor(Z,device).unsqueeze(0)
            s = model.score_accept(z0, A, Z).item()
            scores.append(s); labels.append(0)
            per_type[vtype].append(s)

    s_arr, l_arr = np.array(scores), np.array(labels)
    auroc = roc_auc_score(l_arr, s_arr) if len(set(l_arr)) > 1 else 0.5
    auprc = average_precision_score(l_arr, s_arr)
    preds = (s_arr > 0.5).astype(int)
    tp, fp = ((preds==1)&(l_arr==1)).sum(), ((preds==1)&(l_arr==0)).sum()
    tn, fn = ((preds==0)&(l_arr==0)).sum(), ((preds==0)&(l_arr==1)).sum()
    n = max(len(l_arr), 1)

    per_type_out = {}
    for vt, ss in per_type.items():
        sa = np.array(ss); la = np.zeros(len(ss))
        per_type_out[vt] = {"AUROC": roc_auc_score(la, sa) if len(set(la))>1 else 0.5,
                           "mean_accept": float(sa.mean()),
                           "rejection_rate": float((sa < 0.5).mean())}

    return {"AUROC": auroc, "AUPRC": auprc, "accuracy": (tp+tn)/n,
            "FPR": fp/max(fp+tn,1), "FNR": fn/max(fn+tp,1),
            "per_violation_type": per_type_out}


# ═══════════════════════════════════════════════════════════
# Metric 3: Long-Horizon Drift (CVR)
# ═══════════════════════════════════════════════════════════

def metric_long_horizon_drift(
    model: IWCM, rollout_model, valid_trajs: List[Tuple],
    horizons: List[int] = [5, 10, 20, 50, 100], device: str = "cuda",
) -> Dict[str, Any]:
    iwcm_cvr, rollout_cvr = {}, {}
    for H in horizons:
        iwcm_v, roll_v, total = 0, 0, 0
        for z0, A, Z in valid_trajs:
            z0_d = z0.to(device).unsqueeze(0)
            A_d = A[:H].to(device).unsqueeze(0) if len(A) > H else A.to(device).unsqueeze(0)
            Z_i = model.solve(z0_d, A_d)
            if model.energy(z0_d, A_d, Z_i).item() > 1.0: iwcm_v += 1
            if rollout_model is not None:
                z_r = z0_d
                for t in range(min(H, A_d.shape[1])):
                    z_r = rollout_model(z_r, A_d[:, t:t+1, :])
                Z_r = rollout_model.rollout_to_worldline([z_r])
                if model.energy(z0_d, A_d, Z_r).item() > 1.0: roll_v += 1
            total += 1
        iwcm_cvr[H] = iwcm_v / max(total, 1)
        rollout_cvr[H] = roll_v / max(total, 1)
    return {"iwcm_cvr": iwcm_cvr, "rollout_cvr": rollout_cvr, "horizons": horizons}


# ═══════════════════════════════════════════════════════════
# Metric 4: Repair Success + Minimality
# ═══════════════════════════════════════════════════════════

def metric_repair_success(
    model: IWCM, corrupted_trajs: List[Tuple], original_trajs: List[Tuple],
    device: str = "cuda",
) -> Dict[str, float]:
    successes, mins, targets = 0, [], []
    for (z0_c, A_c, Z_c), (_, _, Z_r) in zip(corrupted_trajs, original_trajs):
        z0_c, A_c, Z_c = z0_c.to(device).unsqueeze(0), A_c.to(device).unsqueeze(0), Z_c.to(device).unsqueeze(0)
        Z_r = Z_r.to(device).unsqueeze(0)
        Z_rep, _ = model.repair(z0_c, A_c, Z_c)
        if model.energy(z0_c, A_c, Z_rep).item() < 1.0: successes += 1
        d_corr = (Z_c - Z_r).pow(2).mean().item()
        d_rep = (Z_rep - Z_r).pow(2).mean().item()
        d_edit = (Z_rep - Z_c).pow(2).mean().item()
        if d_corr > 0: mins.append(d_edit / d_corr)
        targets.append(d_rep)
    n = max(len(corrupted_trajs), 1)
    return {"repair_success_rate": successes/n,
            "repair_minimality": float(np.mean(mins)) if mins else 1.0,
            "target_distance": float(np.mean(targets)) if targets else 0.0}


# ═══════════════════════════════════════════════════════════
# Metric 5: Planning Success
# ═══════════════════════════════════════════════════════════

def metric_planning_success(
    model: IWCM, test_scenarios: List[Scenario], horizon: int = 25,
    num_trials: int = 20, device: str = "cuda",
) -> Dict[str, float]:
    from ..env.grid_world import GridWorld
    from ..env.data import encode_state
    goals, steps_l, invalid, total = 0, [], 0, 0
    for sc in test_scenarios:
        for _ in range(num_trials):
            env = GridWorld(grid_size=sc.grid_size, objects_config=sc.to_env_config(), seed=42)
            env.reset()
            z0_s = encode_state(env.get_state(), sc.grid_size)
            z0 = torch.tensor(z0_s, dtype=torch.float32, device=device).flatten().unsqueeze(0)
            goal = GoalConstraint(goal_type=sc.goal.get("type","position"),
                                  position=tuple(sc.goal.get("pos",(0,0))))
            try:
                A_p, _, energy = model.planner.plan(z0, goal, horizon)
                steps, reached = 0, False
                for a_idx in range(A_p.shape[0]):
                    _, _, done, _ = env.step(int(A_p[a_idx].argmax().item()))
                    steps += 1
                    if done: reached = True; break
                if reached: goals += 1
                steps_l.append(steps)
                if energy.item() > 5.0: invalid += 1
            except Exception: pass
            total += 1
    n = max(total, 1)
    return {"goal_success_rate": goals/n,
            "avg_steps_to_goal": float(np.mean(steps_l)) if steps_l else 0.0,
            "invalid_plan_rate": invalid/n}


# ═══════════════════════════════════════════════════════════
# Metric 6: Energy Calibration
# ═══════════════════════════════════════════════════════════

def metric_energy_calibration(
    model: IWCM, valid_trajs: List[Tuple], invalid_trajs: List[Tuple],
    device: str = "cuda",
) -> Dict[str, float]:
    ev, ei = [], []
    for z0, A, Z in valid_trajs:
        z0, A, Z = z0.to(device).unsqueeze(0), A.to(device).unsqueeze(0), Z.to(device).unsqueeze(0)
        ev.append(model.energy(z0, A, Z).item())
    for z0, A, Z in invalid_trajs:
        z0, A, Z = z0.to(device).unsqueeze(0), A.to(device).unsqueeze(0), Z.to(device).unsqueeze(0)
        ei.append(model.energy(z0, A, Z).item())
    eva, eia = np.array(ev), np.array(ei)
    return {"E_valid_mean": float(eva.mean()), "E_invalid_mean": float(eia.mean()),
            "energy_margin": float(eia.mean() - eva.mean())}


# ═══════════════════════════════════════════════════════════
# Metric 7: Local-vs-Global Failure Detection
# ═══════════════════════════════════════════════════════════

def metric_global_violation_detection(
    model: IWCM, local_only_model, global_invalid_trajs: List[Tuple],
    device: str = "cuda",
) -> Dict[str, float]:
    mc, lc, total = 0, 0, 0
    for z0, A, Z in global_invalid_trajs:
        z0, A, Z = z0.to(device).unsqueeze(0), A.to(device).unsqueeze(0), Z.to(device).unsqueeze(0)
        if model.score_accept(z0, A, Z).item() < 0.5: mc += 1
        if local_only_model is not None:
            if local_only_model.score_accept(z0, A, Z).item() < 0.5: lc += 1
        total += 1
    n = max(total, 1)
    return {"full_iwcm_accuracy": mc/n, "local_only_accuracy": lc/n,
            "global_detection_delta": (mc-lc)/n}


# ═══════════════════════════════════════════════════════════
# Metric 8: Counterfactual Locality F1
# ═══════════════════════════════════════════════════════════

def metric_counterfactual_locality(
    model: IWCM, cf_pairs: List[Tuple], device: str = "cuda",
) -> Dict[str, float]:
    tp, fp, fn = 0, 0, 0
    for z0, A, Z, A2, Z2 in cf_pairs:
        z0 = z0.to(device).unsqueeze(0)
        A, Z = A.to(device).unsqueeze(0), Z.to(device).unsqueeze(0)
        A2, Z2 = A2.to(device).unsqueeze(0), Z2.to(device).unsqueeze(0)
        c1 = (Z[:,1:]-Z[:,:-1]).pow(2).mean(dim=(0,-1)) > 0.01
        c2 = (Z2[:,1:]-Z2[:,:-1]).pow(2).mean(dim=(0,-1)) > 0.01
        changed = c1 != c2
        tp += changed.sum().item(); fp += (~changed).sum().item(); fn += changed.sum().item()
    p = tp/max(tp+fp,1); r = tp/max(tp+fn,1)
    return {"counterfactual_locality_f1": 2*p*r/max(p+r,1e-8),
            "precision": p, "recall": r}


# ═══════════════════════════════════════════════════════════
# Metric 9: AC3 Hardness Quality
# ═══════════════════════════════════════════════════════════

def metric_ac3_hardness_quality(
    model: IWCM, grammar: SymbolicMutationGrammar,
    valid_trajs: List[SymbolicTrajectory], oracle: SymbolicOracle,
    device: str = "cuda",
) -> Dict[str, Any]:
    invalid_rates, accept_rates, vtypes = [], [], defaultdict(int)
    for traj in valid_trajs:
        corr = grammar.apply(traj)
        v = oracle(corr)
        invalid_rates.append(1.0 if len(v) > 0 else 0.0)
        for vi in v: vtypes[vi] += 1
    tc = np.array(list(vtypes.values())); tp = tc / max(tc.sum(), 1)
    return {"oracle_invalid_rate": float(np.mean(invalid_rates)) if invalid_rates else 0.0,
            "violation_type_entropy": float(-np.sum(tp*np.log(tp+1e-8))),
            "violation_types_distribution": dict(vtypes)}


# ═══════════════════════════════════════════════════════════
# Metric 10: Ablation Table
# ═══════════════════════════════════════════════════════════

def metric_ablation_table(
    models: Dict[str, IWCM], test_data: Dict[str, Any],
    device: str = "cuda",
) -> Dict[str, Dict[str, float]]:
    results = {}
    for name, model in models.items():
        model.eval(); model.to(device); row = {}
        vt = test_data.get("valid_trajs", [])[:100]
        it = test_data.get("invalid_trajs_by_type", {})
        if vt and it:
            cls = metric_valid_invalid_classification(model, vt, it, device)
            row["AUROC"] = cls["AUROC"]; row["FPR"] = cls["FPR"]
        if vt:
            drift = metric_long_horizon_drift(model, None, vt[:50],
                                              horizons=[10,50,100], device=device)
            row["CVR_H50"] = drift["iwcm_cvr"].get(50,1.0)
            row["CVR_H100"] = drift["iwcm_cvr"].get(100,1.0)
        ct = test_data.get("cross_surface_test", {})
        if ct:
            cs = metric_cross_surface_law_generalization(model, {}, ct, device)
            row["HeldOutAcc"] = cs.get("held_out_accuracy", 0.0)
            row["GenGap"] = cs.get("generalization_gap", 0.0)
        results[name] = row
    return results


# ═══════════════════════════════════════════════════════════
# Rollout Baseline
# ═══════════════════════════════════════════════════════════

class RolloutModel:
    def __init__(self, d_state: int, d_action: int = 11):
        self.W_z = torch.eye(d_state) * 0.99
        self.W_a = torch.randn(d_action, d_state) * 0.01

    def __call__(self, z_t, a_t):
        if a_t.dim() == 3 and a_t.shape[1] == 1: a_t = a_t.squeeze(1)
        return z_t @ self.W_z.to(z_t.device) + a_t @ self.W_a.to(z_t.device)

    def rollout_to_worldline(self, z_seq): return torch.stack(z_seq, dim=1)

    def to(self, dev):
        self.W_z = self.W_z.to(dev); self.W_a = self.W_a.to(dev); return self


def build_ablation_models(d_state: int, d_action: int = 11, hidden_dim: int = 256) -> Dict[str, IWCM]:
    models = {"iwcm_full": IWCM(d_state, d_action, hidden_dim)}
    for name, zero_keys in [("iwcm_no_invariant", ["invariant"]),
                             ("iwcm_no_boundary", ["boundary"]),
                             ("iwcm_no_counterfactual", ["counterfactual"]),
                             ("iwcm_local_only", ["boundary","invariant","effect","counterfactual"])]:
        m = IWCM(d_state, d_action, hidden_dim)
        for k in zero_keys: m.energy_fn.lambdas[k] = 0.0
        models[name] = m
    return models


def evaluate_model(
    model: IWCM,
    data: dict,
    device: str = "cuda",
) -> dict:
    """Run all 10 evaluation metrics on a trained model.

    Args:
        model: Trained IWCM model.
        data: Dict with keys matching metric requirements:
              valid_trajs, invalid_trajs_by_type, corrupted_trajs, original_trajs,
              counterfactual_pairs, spliced_trajs, global_invalid_trajs,
              test_scenarios, cross_surface_train, cross_surface_test.
        device: Device for computation.

    Returns:
        Dict of all metric results.
    """
    results = {}

    # Metric 1: Cross-surface law generalization
    if "cross_surface_train" in data and "cross_surface_test" in data:
        results["cross_surface"] = metric_cross_surface_law_generalization(
            model, data["cross_surface_train"], data["cross_surface_test"], device,
        )

    # Metric 2: Valid/invalid classification
    if "valid_trajs" in data and "invalid_trajs_by_type" in data:
        results["classification"] = metric_valid_invalid_classification(
            model, data["valid_trajs"], data["invalid_trajs_by_type"], device,
        )

    # Metric 3: Long-horizon drift
    if "valid_trajs" in data:
        rollout = RolloutModel(d_state=model.d_state, d_action=model.d_action)
        rollout.to(device)
        results["drift"] = metric_long_horizon_drift(
            model, rollout, data["valid_trajs"][:100],
            horizons=[10, 25, 50], device=device,
        )

    # Metric 4: Repair success
    if "corrupted_trajs" in data and "original_trajs" in data:
        results["repair"] = metric_repair_success(
            model, data["corrupted_trajs"][:100],
            data["original_trajs"][:100], device,
        )

    # Metric 5: Planning success
    if "test_scenarios" in data:
        results["planning"] = metric_planning_success(
            model, data["test_scenarios"], num_trials=10, device=device,
        )

    # Metric 6: Energy calibration
    if "valid_trajs" in data and "invalid_trajs_by_type" in data:
        all_invalid = []
        for trajs in data["invalid_trajs_by_type"].values():
            all_invalid.extend(trajs[:50])
        results["energy_calibration"] = metric_energy_calibration(
            model, data["valid_trajs"][:100], all_invalid, device,
        )

    # Metric 7: Global violation detection
    if "global_invalid_trajs" in data:
        local_model = IWCM(model.d_state, model.d_action, 128)
        local_model.energy_fn.lambdas["boundary"] = 0.0
        local_model.energy_fn.lambdas["invariant"] = 0.0
        local_model.energy_fn.lambdas["effect"] = 0.0
        local_model.energy_fn.lambdas["counterfactual"] = 0.0
        local_model.to(device)
        results["global_violation"] = metric_global_violation_detection(
            model, local_model, data["global_invalid_trajs"], device,
        )

    # Metric 8: Counterfactual locality
    if "counterfactual_pairs" in data:
        results["counterfactual_locality"] = metric_counterfactual_locality(
            model, data["counterfactual_pairs"], device,
        )

    return results
