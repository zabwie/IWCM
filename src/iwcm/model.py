"""IWCM model wrapper — integrates energy, solver, refinement, and planner.

Provides a single interface for the complete IWCM system:
- energy computation
- worldline solving/refinement
- planning
- repair (fix corrupted worldlines)
- acceptance scoring
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple

from .energy import IWCMEnergy
from .solver import GradientDescentSolver
from .refinement import LearnedRefinementOperator, RefinementSolver
from .planner import IWCMPlanner, GoalConstraint
from ..utils.base import BaseModel


class IWCM(BaseModel):
    """Complete Implicit Worldline Constraint Model.

    Wraps energy function, solver, refinement operator, and planner
    into a single trainable model.

    Usage:
        model = IWCM(d_state=256, d_action=11, hidden_dim=256)
        energy = model.energy(z0, A, Z)
        Z_solved = model.solve(z0, A)
        repaired = model.repair(z0, A, corrupted_Z)
        accept = model.score_accept(z0, A, Z)
        A, Z = model.plan(z0, goal, horizon=25)
    """

    def __init__(
        self,
        d_state: int,
        d_action: int = 11,
        hidden_dim: int = 256,
        lambdas: Optional[Dict[str, float]] = None,
        solver_steps: int = 20,
        solver_lr: float = 0.01,
        use_refinement: bool = True,
    ):
        super().__init__()

        # Energy function
        self.energy_fn = IWCMEnergy(d_state, d_action, hidden_dim, lambdas)

        # Gradient descent solver
        self.gd_solver = GradientDescentSolver(
            self.energy_fn,
            num_steps=solver_steps,
            lr=solver_lr,
        )

        # Learned refinement (optional)
        self.ref_op = None
        if use_refinement:
            self.ref_op = LearnedRefinementOperator(d_state, d_action, hidden_dim)

        self.hybrid_solver = RefinementSolver(
            self.energy_fn,
            gd_steps=solver_steps // 2,
            gd_lr=solver_lr,
            ref_op=self.ref_op,
            ref_steps=3,
        )

        # Planner
        self.planner = IWCMPlanner(self.energy_fn, self.gd_solver, d_action)

        self.d_state = d_state
        self.d_action = d_action
        self.hidden_dim = hidden_dim

    def energy(
        self, z0: torch.Tensor, A: torch.Tensor, Z: torch.Tensor
    ) -> torch.Tensor:
        """Compute total IWCM energy for a worldline.

        Args:
            z0: (B, d_state).
            A: (B, H, d_action).
            Z: (B, H, d_state).

        Returns:
            Energy per batch, shape (B,).
        """
        return self.energy_fn(z0, A, Z)

    def energy_per_head(
        self, z0: torch.Tensor, A: torch.Tensor, Z: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Compute per-constraint-head energy breakdown."""
        return self.energy_fn.compute_per_head(z0, A, Z)

    def solve(
        self, z0: torch.Tensor, A: torch.Tensor, use_hybrid: bool = True
    ) -> torch.Tensor:
        """Solve for optimal worldline Z given z0 and actions A.

        Args:
            z0: (B, d_state).
            A: (B, H, d_action).
            use_hybrid: Use hybrid (GD + learned) solver if available.

        Returns:
            Refined Z, shape (B, H, d_state).
        """
        if use_hybrid and self.ref_op is not None:
            return self.hybrid_solver.solve(z0, A)
        return self.gd_solver(z0, A)

    def score_accept(
        self, z0: torch.Tensor, A: torch.Tensor, Z: torch.Tensor
    ) -> torch.Tensor:
        """Compute acceptance score ∈ [0, 1] from energy."""
        return self.energy_fn.score_acceptance(z0, A, Z)

    def repair(
        self, z0: torch.Tensor, A: torch.Tensor, corrupted_Z: torch.Tensor,
        num_steps: int = 30
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Repair a corrupted worldline by minimizing energy from corrupted init.

        Args:
            z0: (B, d_state).
            A: (B, H, d_action).
            corrupted_Z: (B, H, d_state) — causally invalid worldline.
            num_steps: Extra solver steps for repair.

        Returns:
            repaired_Z: (B, H, d_state).
            energy_improvement: (B,) — difference in energy before/after.
        """
        with torch.no_grad():
            energy_before = self.energy_fn(z0, A, corrupted_Z)

        solver = GradientDescentSolver(
            self.energy_fn, num_steps=num_steps, lr=0.005
        )
        repaired_Z, _ = solver.solve(z0, A, init_Z=corrupted_Z)

        with torch.no_grad():
            energy_after = self.energy_fn(z0, A, repaired_Z)

        improvement = energy_before - energy_after
        return repaired_Z, improvement

    def plan(
        self, z0: torch.Tensor, goal: GoalConstraint, horizon: int = 25
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Plan optimal actions and worldline to reach goal.

        Args:
            z0: (d_state,) or (B, d_state).
            goal: Goal specification.
            horizon: Planning horizon.

        Returns:
            (A, Z) for the best plan.
        """
        return self.planner(z0, goal, horizon)

    def forward(
        self, z0: torch.Tensor, A: torch.Tensor, Z: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Full forward pass: energy + per-head breakdown.

        Returns:
            Dict with "energy" and per-head scores.
        """
        return {
            "energy": self.energy(z0, A, Z),
            **self.energy_per_head(z0, A, Z),
        }
