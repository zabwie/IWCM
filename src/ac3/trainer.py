"""AC3 Training Loop (Algorithm 1 from paper).

Three-system co-evolutionary training:
  1. World Model W_θ — learns low energy for valid, high for invalid, repair
  2. Corruptor C_φ — generates near-miss worlds
  3. Constraint Oracle O — determines causal validity
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional
from torch.utils.data import DataLoader
import numpy as np

from .corruptor import AC3Corruptor
from .oracle import SymbolicOracle
from .hardness import HardnessScorer, CurriculumManager
from .mutations.grammar import SymbolicMutationGrammar, SymbolicTrajectory
from ..iwcm.model import IWCM
from ..env.symbolic_state import SymbolicState, encode_symbolic_trajectory, symbolic_to_state_dict
from ..env.data import encode_state, encode_action
from ..utils.logging import MetricsLogger


def _encode_corrupted_batch(
    corrupted_trajs: List[SymbolicTrajectory],
    grid_size: int = 8,
    horizon: int = 25,
    device: str = "cuda",
) -> tuple:
    """Encode corrupted symbolic trajectories into (z0, A, Z) tensors."""
    z0_list, A_list, Z_list = [], [], []
    d_state = grid_size * grid_size * 4
    for ct in corrupted_trajs:
        enc = encode_symbolic_trajectory(ct, grid_size, horizon)
        if enc is None:
            continue
        z0_np, A_np, Z_np = enc
        z0_list.append(torch.from_numpy(z0_np).reshape(d_state))
        A_list.append(torch.from_numpy(A_np))
        Z_list.append(torch.from_numpy(Z_np).reshape(horizon, d_state))
    if not z0_list:
        return None, None, None
    return (
        torch.stack(z0_list).to(device),
        torch.stack(A_list).to(device),
        torch.stack(Z_list).to(device),
    )


def _make_symbolic_from_tensor(
    z0: torch.Tensor, A: torch.Tensor, Z: torch.Tensor, grid_size: int = 8,
) -> List[SymbolicTrajectory]:
    """Reconstruct symbolic trajectories from tensor data using a best-effort decode."""
    from .mutations.grammar import SymbolicTrajectory as ST
    B, H, d = Z.shape
    trajs = []
    for b in range(B):
        actions = A[b].argmax(dim=-1).cpu().tolist()
        # Try to reconstruct state from tensor (heuristic)
        z0_flat = z0[b].reshape(grid_size, grid_size, 4).cpu().numpy()
        agent_r = int(np.argmax(z0_flat[:, :, 0]))
        agent_c = agent_r % grid_size
        agent_r //= grid_size
        if agent_r >= grid_size:
            agent_r, agent_c = 0, 0

        sym_states = []
        for t in range(H + 1):
            z_flat = (z0_flat if t == 0 else Z[b, t-1].reshape(grid_size, grid_size, 4).cpu().numpy())
            obj_positions, obj_types = {}, {}
            for r in range(grid_size):
                for c in range(grid_size):
                    obj_type_val = z_flat[r, c, 1]
                    if obj_type_val > 0.1:
                        oid = f"obj_{r}_{c}"
                        type_map = {0.0: "unknown", 1.0: "key", 2.0: "door", 3.0: "box", 4.0: "occluder"}
                        ot = min(type_map.keys(), key=lambda k: abs(k - obj_type_val))
                        obj_positions[oid] = (r, c)
                        obj_types[oid] = type_map.get(ot, "unknown")
            sym_states.append(SymbolicState(
                agent_pos=(agent_r, agent_c), object_positions=obj_positions,
                object_types=obj_types, grid_size=grid_size, step=t,
            ))
        trajs.append(ST(states=sym_states, actions=actions, horizon=H))
    return trajs


class AC3Trainer:
    def __init__(self, world_model: IWCM, corruptor: AC3Corruptor, oracle: SymbolicOracle,
                 logger=None, device: str = "cuda", lr_world: float = 1e-4,
                 lr_corruptor: float = 1e-4, num_mutations_per_sample: int = 4,
                 top_k_hard: int = 8, accept_low: float = 0.4, accept_high: float = 0.7,
                 lambda_valid: float = 1.0, lambda_invalid: float = 1.0,
                 lambda_repair: float = 0.5, lambda_accept: float = 0.5,
                 lambda_minimal: float = 0.3, lambda_diversity: float = 0.2,
                 grid_size: int = 8, horizon: int = 25):
        self.world_model = world_model.to(device)
        self.corruptor = corruptor.to(device)
        self.oracle = oracle
        self.logger = logger
        self.device = device
        self.grid_size = grid_size
        self.horizon = horizon

        self.opt_world = torch.optim.Adam(self.world_model.parameters(), lr=lr_world)
        self.opt_corruptor = torch.optim.Adam(self.corruptor.parameters(), lr=lr_corruptor)

        self.hardness_scorer = HardnessScorer()
        self.curriculum = CurriculumManager(accept_low=accept_low, accept_high=accept_high, top_k=top_k_hard)

        self.M = num_mutations_per_sample
        self.lambda_valid = lambda_valid
        self.lambda_invalid = lambda_invalid
        self.lambda_repair = lambda_repair
        self.lambda_accept = lambda_accept
        self.lambda_minimal = lambda_minimal
        self.lambda_diversity = lambda_diversity
        self.global_step = 0

    def train_step(self, z0, A, Z, symbolic_trajs):
        B = z0.shape[0]
        device = self.device
        gs, H = self.grid_size, self.horizon
        d_state = gs * gs * 4

        # Phase 1: Generate diverse corrupted symbolic trajectories
        # Apply ALL mutation types per trajectory for cross-surface coverage
        all_corrupted: List[SymbolicTrajectory] = []
        for traj in symbolic_trajs:
            for mutation in self.corruptor.grammar.mutations:
                corrupted = mutation.apply(traj, np.random.RandomState())
                all_corrupted.append(corrupted)

        # Phase 1b: ENCODE corrupted trajectories into actual Z tensors
        z0_corr, A_corr, Z_corr = _encode_corrupted_batch(
            all_corrupted, gs, H, device,
        )
        if z0_corr is None:
            return {"loss_world": 0.0, "loss_corruptor": 0.0, "energy_valid": 0.0,
                    "energy_invalid": 0.0, "repair_improvement": 0.0,
                    "accept_mean": 0.5, "violation_rate": 0.0, "curriculum_mean_accept": 0.5}

        C = z0_corr.shape[0]  # number of successfully encoded corruptions

        # Phase 2: Oracle evaluation
        violation_counts = [len(self.oracle(ct)) for ct in all_corrupted[:C]]
        violation_tensor = torch.tensor(violation_counts, dtype=torch.float32, device=device)

        # Phase 3: World Model training — use hardness-filtered corruptions
        self.opt_world.zero_grad()

        energy_valid = self.world_model.energy(z0, A, Z)
        energy_invalid = self.world_model.energy(z0_corr, A_corr, Z_corr)

        # Compute hardness and select only corruptions near the model's uncertainty boundary
        with torch.no_grad():
            accept_scores_pre = self.world_model.score_accept(z0_corr, A_corr, Z_corr)

        # Select hard negatives: those the model is uncertain about
        in_range = (accept_scores_pre >= self.curriculum.accept_low) & (
            accept_scores_pre <= self.curriculum.accept_high)

        if in_range.sum() > 4:
            # Train only on hard corruptions
            hard_indices = in_range.nonzero(as_tuple=True)[0]
            energy_invalid_hard = self.world_model.energy(
                z0_corr[hard_indices], A_corr[hard_indices], Z_corr[hard_indices],
            )
            loss_invalid = self.lambda_invalid * F.relu(1.0 - energy_invalid_hard).mean()
        else:
            loss_invalid = self.lambda_invalid * F.relu(1.0 - energy_invalid).mean()

        loss_valid = self.lambda_valid * F.relu(energy_valid + 1.0).mean()
        reg = 0.001 * (energy_valid.pow(2).mean() + energy_invalid.pow(2).mean())

        repaired_Z, repair_imp = self.world_model.repair(z0_corr, A_corr, Z_corr)
        loss_repair = self.lambda_repair * (-repair_imp.mean())

        loss_world = loss_valid + loss_invalid + loss_repair + reg
        loss_world.backward()
        torch.nn.utils.clip_grad_norm_(self.world_model.parameters(), 1.0)
        self.opt_world.step()

        # Phase 4: Corruptor training (with hardness selection)
        self.opt_corruptor.zero_grad()
        _, type_probs, _ = self.corruptor(symbolic_trajs)

        with torch.no_grad():
            accept_scores = self.world_model.score_accept(z0_corr, A_corr, Z_corr)

        # Hardness selection: only train on hard negatives
        surface_d = torch.zeros_like(accept_scores)
        edit_d = surface_d
        hardness = self.hardness_scorer.compute(accept_scores, surface_d, edit_d, violation_tensor)
        in_range = (accept_scores >= self.curriculum.accept_low) & (
            accept_scores <= self.curriculum.accept_high)
        if in_range.sum() > 0:
            hard_accept = accept_scores[in_range]
        else:
            hard_accept = accept_scores

        if type_probs.numel() > 0:
            entropy = -(type_probs * torch.log(type_probs + 1e-8)).sum(dim=-1).mean()
            loss_corruptor = -(self.lambda_accept * hard_accept.mean() +
                              self.lambda_diversity * entropy)
        else:
            loss_corruptor = -self.lambda_accept * hard_accept.mean()

        loss_corruptor.backward()
        self.opt_corruptor.step()

        self.curriculum.update(accept_scores)
        self.global_step += 1

        metrics = {
            "loss_world": loss_world.item(),
            "loss_corruptor": loss_corruptor.item(),
            "energy_valid": energy_valid.mean().item(),
            "energy_invalid": energy_invalid.mean().item(),
            "repair_improvement": repair_imp.mean().item(),
            "accept_mean": accept_scores.mean().item(),
            "violation_rate": violation_tensor.float().mean().item(),
            "curriculum_mean_accept": self.curriculum.get_mean_acceptance(),
        }
        if self.logger:
            self.logger.log_metrics(metrics, self.global_step)
        return metrics

    def train_epoch(self, dataloader):
        epoch_metrics = []
        for batch in dataloader:
            if len(batch) == 3:
                z0, A, Z = batch
                symbolic = None
            else:
                z0, A, Z, symbolic = batch
            z0, A, Z = z0.to(self.device), A.to(self.device), Z.to(self.device)
            if symbolic is None:
                symbolic = _make_symbolic_from_tensor(z0, A, Z, self.grid_size)
            m = self.train_step(z0, A, Z, symbolic)
            epoch_metrics.append(m)
        avg = {}
        for key in epoch_metrics[0]:
            avg[key] = np.mean([m[key] for m in epoch_metrics])
        return avg
