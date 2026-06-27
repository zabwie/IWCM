"""IWCM gradient descent solver.

Iteratively refines worldline Z to minimize the energy function:
  Z^{(k+1)} = Z^{(k)} - α · ∇_Z E_θ(z0, A, Z^{(k)})

Initialize Z^{(0)} ~ N(0, I), then refine for K steps.
Optimized for GPU with batched gradient computation.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple

from .energy import IWCMEnergy


class GradientDescentSolver(nn.Module):
    """Gradient-based worldline refinement solver.

    Finds Z that minimizes E_θ(z0, A, Z) via gradient descent.
    This breaks the autoregressive chain: all z_t are free variables
    optimized jointly, eliminating structural drift.

    Args:
        energy_fn: The IWCM energy function.
        num_steps: Number of gradient steps (default: 20).
        lr: Learning rate for gradient descent (default: 0.01).
        momentum: Momentum factor (0 = no momentum).
    """

    def __init__(
        self,
        energy_fn: IWCMEnergy,
        num_steps: int = 20,
        lr: float = 0.01,
        momentum: float = 0.9,
    ):
        super().__init__()
        self.energy_fn = energy_fn
        self.num_steps = num_steps
        self.lr = lr
        self.momentum = momentum

    def solve(
        self,
        z0: torch.Tensor,
        A: torch.Tensor,
        init_Z: Optional[torch.Tensor] = None,
        track_energy: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Refine worldline Z to minimize energy.

        Args:
            z0: Initial state encoding, shape (B, d_state).
            A: Action sequence, shape (B, H, d_action).
            init_Z: Optional initial guess. If None, sampled from N(0, 1).
            track_energy: If True, return energy trajectory.

        Returns:
            Z: Refined worldline, shape (B, H, d_state).
            energy_trace: (num_steps+1, B) if track_energy else None.
        """
        B = z0.shape[0]
        H = A.shape[1]
        d_state = self.energy_fn.d_state
        device = z0.device

        # Initialize Z ~ N(0, I)
        if init_Z is not None:
            Z = init_Z.clone().detach().requires_grad_(True)
        else:
            Z = torch.randn(B, H, d_state, device=device, requires_grad=True)

        energy_trace = None
        if track_energy:
            energy_trace = []

        velocity = torch.zeros_like(Z)

        for step in range(self.num_steps):
            if track_energy:
                with torch.no_grad():
                    e = self.energy_fn(z0, A, Z)
                    energy_trace.append(e.clone())

            # Compute gradient
            energy = self.energy_fn(z0, A, Z).mean()
            grad = torch.autograd.grad(energy, Z, create_graph=False)[0]

            # Momentum update
            velocity = self.momentum * velocity + grad
            Z = Z - self.lr * velocity

            # Re-detach for next iteration (optional: keep graph for learned lr)
            Z = Z.detach().requires_grad_(True)
            velocity = velocity.detach()

        if track_energy:
            with torch.no_grad():
                e = self.energy_fn(z0, A, Z)
                energy_trace.append(e.clone())
            energy_trace = torch.stack(energy_trace, dim=0)  # (steps+1, B)

        return Z, energy_trace

    def forward(
        self, z0: torch.Tensor, A: torch.Tensor
    ) -> torch.Tensor:
        """Convenience: solve and return Z."""
        Z, _ = self.solve(z0, A)
        return Z
