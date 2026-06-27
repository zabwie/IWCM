"""AC3 Corruptor C_φ — learned adversarial near-miss generator.

Generates causally invalid but surface-plausible trajectories by
selecting and parameterizing symbolic mutations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Optional, Tuple

from .mutations.grammar import SymbolicMutationGrammar, SymbolicTrajectory
from ..env.symbolic_state import SymbolicState
from ..utils.base import BaseModel


class AC3Corruptor(BaseModel):
    """Learned corruptor for AC3 adversarial training.

    Selects which mutation types to apply and with what parameters
    to generate near-miss worlds that exploit model weaknesses.

    For symbolic environments (Experiment 1), the corruptor operates
    on symbolic trajectories. For continuous environments (Experiment 2),
    the TAMG corruptor replaces this.
    """

    def __init__(
        self,
        d_state: int,
        num_mutation_types: int = 7,
        hidden_dim: int = 128,
    ):
        super().__init__()
        self.d_state = d_state
        self.num_mutation_types = num_mutation_types

        # Mutation type selector: given encoded trajectory, predict which
        # mutation types to apply
        self.type_selector = nn.Sequential(
            nn.Linear(d_state * 2, hidden_dim),  # encode start+end state
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_mutation_types),
        )

        # Edit magnitude predictor: how much to corrupt
        self.magnitude_predictor = nn.Sequential(
            nn.Linear(d_state * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),  # ∈ [0, 1]
        )

        # Grammar (symbolic mutation operators)
        self.grammar = SymbolicMutationGrammar()

    def _encode_trajectory(
        self, traj: SymbolicTrajectory
    ) -> torch.Tensor:
        device = next(self.type_selector.parameters()).device

        def _state_vector(state: SymbolicState) -> torch.Tensor:
            vec = torch.zeros(self.d_state, device=device)
            idx = 0
            if state.agent_pos:
                vec[idx] = float(state.agent_pos[0]) / 10.0
                vec[idx + 1] = float(state.agent_pos[1]) / 10.0
            idx += 2
            for t in ["key", "door", "box", "occluder"]:
                count = sum(1 for o in state.object_types.values() if o == t)
                if idx < self.d_state: vec[idx] = float(count) / 10.0
                idx += 1
            if idx < self.d_state: vec[idx] = float(sum(state.door_states.values())) / 10.0
            idx += 1
            if idx < self.d_state: vec[idx] = float(len(state.inventory)) / 5.0
            return vec

        start_vec = _state_vector(traj.states[0])
        end_vec = _state_vector(traj.states[-1])
        return torch.cat([start_vec, end_vec], dim=-1)

    def forward(
        self, trajectories: List[SymbolicTrajectory]
    ) -> Tuple[List[SymbolicTrajectory], torch.Tensor, torch.Tensor]:
        """Generate corrupted versions of input trajectories.

        Args:
            trajectories: List of valid symbolic trajectories.

        Returns:
            corrupted: List of corrupted trajectories.
            type_probs: Mutation type probabilities, shape (B, 7).
            magnitudes: Corruption magnitude per sample, shape (B,).
        """
        rng = np.random.RandomState()
        corrupted: List[SymbolicTrajectory] = []
        type_probs_list = []
        magnitudes_list = []

        for traj in trajectories:
            encoding = self._encode_trajectory(traj).unsqueeze(0)  # (1, d*2)

            # Predict mutation type distribution
            logits = self.type_selector(encoding).squeeze(0)  # (7,)
            probs = F.softmax(logits, dim=-1)
            type_probs_list.append(probs)

            # Predict corruption magnitude
            magnitude = self.magnitude_predictor(encoding).squeeze(0)  # (1,)

            # Apply selected mutation
            mutation_idx = torch.multinomial(probs, 1).item()
            mutation = self.grammar.mutations[mutation_idx]

            # Scale the mutation severity based on predicted magnitude
            traj_corrupted = mutation.apply(traj, rng)
            corrupted.append(traj_corrupted)
            magnitudes_list.append(magnitude)

        device = next(self.type_selector.parameters()).device
        type_probs = torch.stack(type_probs_list) if type_probs_list else torch.empty(0, 7, device=device)
        magnitudes = torch.stack(magnitudes_list) if magnitudes_list else torch.empty(0, device=device)

        return corrupted, type_probs, magnitudes

    def generate_batch(
        self, trajectories: List[SymbolicTrajectory],
        num_per_sample: int = 4,
    ) -> Tuple[List[SymbolicTrajectory], List[SymbolicTrajectory]]:
        """Generate multiple corruptions for each training sample.

        Returns tuple of (originals, corrupted) with corrupted
        being num_per_sample times longer than originals.
        """
        all_originals = []
        all_corrupted = []

        for traj in trajectories:
            for _ in range(num_per_sample):
                all_originals.append(traj)
                corrupted_batch, _, _ = self.forward([traj])
                all_corrupted.append(corrupted_batch[0])

        return all_originals, all_corrupted
