"""IWCM constraint heads — base class and registry."""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional
from src.utils.base import BaseModel


class ConstraintHead(BaseModel):
    """Base class for IWCM constraint heads.

    Each head takes (z0, A, Z) and outputs a scalar violation score.
    Subclasses implement different constraint types.

    Input shapes:
      z0: (B, d_state) — encoded initial state
      A:  (B, H, d_action) — action embeddings over horizon
      Z:  (B, H, d_state) — encoded state sequence over horizon
    """

    def __init__(self, d_state: int, d_action: int = 11, hidden_dim: int = 256):
        super().__init__()
        self.d_state = d_state
        self.d_action = d_action
        self.hidden_dim = hidden_dim

    def forward(
        self, z0: torch.Tensor, A: torch.Tensor, Z: torch.Tensor
    ) -> torch.Tensor:
        """Compute constraint violation score.

        Args:
            z0: Initial state encoding, shape (B, d_state).
            A: Action sequence encoding, shape (B, H, d_action).
            Z: State sequence encoding, shape (B, H, d_state).

        Returns:
            Scalar violation score per batch element, shape (B,).
        """
        raise NotImplementedError


class ConstraintRegistry:
    """Registry for named constraint heads with configurable weights."""

    def __init__(self):
        self._heads: Dict[str, ConstraintHead] = {}
        self._weights: Dict[str, float] = {}

    def register(self, name: str, head: ConstraintHead, weight: float = 1.0) -> None:
        self._heads[name] = head
        self._weights[name] = weight

    def get(self, name: str) -> ConstraintHead:
        return self._heads[name]

    @property
    def head_names(self) -> List[str]:
        return list(self._heads.keys())

    def compute_all(
        self, z0: torch.Tensor, A: torch.Tensor, Z: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Compute weighted sum of all constraint violations.

        Args:
            z0, A, Z: Standard constraint head inputs.

        Returns:
            total_energy: shape (B,) — weighted sum of all head outputs.
            per_head: dict mapping head name to raw violation score.
        """
        total = torch.zeros(z0.shape[0], device=z0.device)
        per_head: Dict[str, torch.Tensor] = {}

        for name in self._heads:
            score = self._heads[name](z0, A, Z)
            weight = self._weights[name]
            total = total + weight * score
            per_head[name] = score.detach()

        return total, per_head
