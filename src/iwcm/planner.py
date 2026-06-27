"""IWCM Planner — MAP inference for action and state optimization.

Planning as MAP inference:
  (A*, Z*) = argmin_{A,Z} E_θ(z0, A, Z) + C_goal(Z)

Optimizes over both action sequences and state worldlines simultaneously,
without autoregressive rollout.
"""

import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional

from .energy import IWCMEnergy
from .solver import GradientDescentSolver


class GoalConstraint:
    """Defines a goal constraint for planning."""

    def __init__(
        self,
        goal_type: str = "position",
        position: Optional[Tuple[int, int]] = None,
        door_id: Optional[str] = None,
        key_id: Optional[str] = None,
        weight: float = 10.0,
    ):
        self.goal_type = goal_type
        self.position = position
        self.door_id = door_id
        self.key_id = key_id
        self.weight = weight

    def to_embedding(self, d_state: int, device: torch.device) -> torch.Tensor:
        """Create a goal embedding for the planner."""
        emb = torch.zeros(d_state, device=device)
        if self.position is not None:
            emb[0] = float(self.position[0]) / 10.0
            emb[1] = float(self.position[1]) / 10.0
        return emb


class IWCMPlanner(nn.Module):
    """IWCM planner for long-horizon planning via constraint satisfaction.

    Instead of autoregressive rollout, jointly optimizes actions A and
    states Z to satisfy constraints and reach the goal.
    """

    def __init__(
        self,
        energy_fn: IWCMEnergy,
        solver: GradientDescentSolver,
        d_action: int = 11,
    ):
        super().__init__()
        self.energy_fn = energy_fn
        self.solver = solver
        self.d_action = d_action

    def plan(
        self,
        z0: torch.Tensor,
        goal: GoalConstraint,
        horizon: int,
        num_candidates: int = 8,
        refine_steps: int = 10,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Plan optimal action sequence and worldline.

        Uses random shooting + refinement: generates N candidate action
        sequences, refines state worldline for each, picks best.

        Args:
            z0: Initial state, shape (d_state,) or (B, d_state).
            goal: Goal specification.
            horizon: Planning horizon.
            num_candidates: Number of action sequences to try.
            refine_steps: Solver steps per candidate.

        Returns:
            A_best: Best action sequence, shape (horizon, d_action).
            Z_best: Best worldline, shape (horizon, d_state).
            energy_best: Energy of best plan.
        """
        single_input = z0.dim() == 1
        if single_input:
            z0 = z0.unsqueeze(0)

        device = z0.device
        d_state = self.energy_fn.d_state
        best_energy = float("inf")
        best_A = None
        best_Z = None

        # Goal embedding
        goal_emb = goal.to_embedding(d_state, device)

        for _ in range(num_candidates):
            # Random action sequence
            A_logits = torch.randn(1, horizon, self.d_action, device=device)
            A = torch.softmax(A_logits, dim=-1)  # (1, H, d_action)

            # Solve for Z given actions
            Z, _ = self.solver.solve(z0, A)

            # Compute total cost = energy + goal cost
            energy = self.energy_fn(z0, A, Z)  # (1,)

            # Goal cost: final state should match goal
            goal_state = Z[:, -1, :]  # last state
            goal_cost = goal.weight * torch.norm(goal_state - goal_emb, dim=-1)

            total = (energy + goal_cost).item()

            if total < best_energy:
                best_energy = total
                best_A = A.detach()
                best_Z = Z.detach()

        if single_input:
            best_A = best_A.squeeze(0)
            best_Z = best_Z.squeeze(0)
            best_energy = torch.tensor(best_energy, device=device)

        return best_A, best_Z, best_energy

    def forward(
        self, z0: torch.Tensor, goal: GoalConstraint, horizon: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Plan and return best (A, Z)."""
        A, Z, _ = self.plan(z0, goal, horizon)
        return A, Z
