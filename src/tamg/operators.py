"""TAMG learned operator basis (Appendix B).

Z' = Z + Σ_k α_k · O_k(Z)

Each operator O_k is a low-rank adapter over slot trajectories
with soft time, slot, and feature masks. Learned from natural
latent differences, then fine-tuned adversarially.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional
from sklearn.cluster import KMeans
import numpy as np

from ..utils.base import BaseModel


class OperatorBasis(BaseModel):
    """Learned basis of causal mutation operators.

    Operators are initialized from clustered latent differences and
    fine-tuned adversarially to produce manifold-preserving corruptions.
    """

    def __init__(
        self,
        num_operators: int = 16,
        slot_dim: int = 64,
        hidden_dim: int = 32,
        rank: int = 8,
    ):
        super().__init__()
        self.num_operators = num_operators
        self.slot_dim = slot_dim
        self.rank = rank

        # Low-rank adapters: A_k · B_k^T where A_k ∈ R^(d×r), B_k ∈ R^(d×r)
        self.A = nn.Parameter(torch.randn(num_operators, slot_dim, rank) * 0.02)
        self.B = nn.Parameter(torch.randn(num_operators, slot_dim, rank) * 0.02)

        # Time mask: which time steps each operator affects
        self.time_mask = nn.Parameter(torch.zeros(num_operators, 1, 1, 1))

        # Slot mask: which slots each operator affects
        self.slot_mask = nn.Parameter(torch.zeros(num_operators, 1, 1, 1))

        # Feature mask: which feature dimensions each operator affects
        self.feat_mask = nn.Parameter(torch.zeros(num_operators, 1, 1, slot_dim))

        # Coefficients predictor (from trajectory encoding)
        self.coef_predictor = nn.Sequential(
            nn.Linear(slot_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_operators),
        )

    def init_from_differences(
        self, diff_vectors: List[torch.Tensor],
    ) -> None:
        """Initialize operators from clustered latent difference vectors.

        Args:
            diff_vectors: List of (..., slot_dim) difference tensors
                representing natural temporal changes in latent space.
        """
        # Flatten all differences
        all_diffs = torch.cat([d.reshape(-1, self.slot_dim) for d in diff_vectors], dim=0)
        diffs_np = all_diffs.detach().cpu().numpy()

        # Cluster into K modes
        n_clusters = min(self.num_operators, len(diffs_np))
        if n_clusters < 2:
            return

        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(diffs_np)

        # Initialize each operator from cluster centroid
        for k in range(n_clusters):
            centroid = torch.tensor(
                kmeans.cluster_centers_[k], dtype=torch.float32,
            )
            # Initialize A_k * B_k^T ≈ centroid via SVD
            U, S, Vt = torch.linalg.svd(centroid.unsqueeze(1), full_matrices=False)
            rank = min(self.rank, len(S))
            with torch.no_grad():
                self.A.data[k, :, :rank] = U[:, :rank] * S[:rank].sqrt()
                self.B.data[k, :, :rank] = Vt[:rank, :].T * S[:rank].sqrt()

    def forward(
        self, Z: torch.Tensor, return_coefficients: bool = False,
    ) -> torch.Tensor:
        """Apply operator basis to worldline.

        Args:
            Z: Worldline slab, shape (B, H, N, d).
            return_coefficients: If True, also return α_k.

        Returns:
            Modified worldline Z', shape (B, H, N, d).
        """
        B, H, N, d = Z.shape
        K = self.num_operators

        # Compute operator matrices: W_k = A_k @ B_k^T
        W = torch.einsum("kdr,krD->kdD", self.A, self.B.transpose(1, 2))  # (K, d, d)

        # Masks (with sigmoid gating)
        t_mask = torch.sigmoid(self.time_mask).expand(K, H, N, 1)  # (K, H, N, 1)
        s_mask = torch.sigmoid(self.slot_mask).expand(K, H, N, 1)
        f_mask = torch.sigmoid(self.feat_mask).expand(K, H, N, d)

        combined_mask = t_mask * s_mask * f_mask  # (K, H, N, d)

        # Compute operator outputs: O_k(Z) = Z @ W_k^T (applied per time/slot)
        Z_expanded = Z.unsqueeze(0).expand(K, B, H, N, d)  # (K, B, H, N, d)
        O_k_Z = torch.einsum("kbhnd,kdw->kbhnw", Z_expanded, W)  # (K, B, H, N, d)

        # Predict coefficients α_k
        # Encode trajectory for coefficient prediction
        z_start = Z[:, 0].mean(dim=1)  # (B, d) — mean over slots at t=0
        z_end = Z[:, -1].mean(dim=1)   # (B, d) — mean over slots at t=H-1
        traj_enc = torch.cat([z_start, z_end], dim=-1)  # (B, 2d)
        alpha = self.coef_predictor(traj_enc)  # (B, K)
        alpha = torch.tanh(alpha) * 0.1  # small coefficients

        # Combine: Z' = Z + Σ_k α_k · mask_k ⊙ O_k(Z)
        alpha_expanded = alpha.T.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)  # (K, B, 1, 1, 1)
        mask_reshaped = combined_mask.unsqueeze(1)  # (K, 1, H, N, d)

        delta = (alpha_expanded * mask_reshaped * O_k_Z).sum(dim=0)  # (B, H, N, d)
        Z_prime = Z + delta

        if return_coefficients:
            return Z_prime, alpha
        return Z_prime


def learn_operator_basis(
    latent_differences: List[torch.Tensor],
    num_operators: int = 16,
    slot_dim: int = 64,
    num_clusters: int = 16,
) -> OperatorBasis:
    """Initialize operator basis from latent differences.

    Collects v_{i,t} = z_{i,t+1} - z_{i,t} from encoded trajectories,
    clusters them, and initializes operators from cluster centroids.

    Args:
        latent_differences: List of difference tensors from encoder.
        num_operators: Number of operators in the basis.
        slot_dim: Dimension per slot.
        num_clusters: Number of clusters for initialization.

    Returns:
        Initialized OperatorBasis.
    """
    basis = OperatorBasis(
        num_operators=num_operators,
        slot_dim=slot_dim,
    )
    basis.init_from_differences(latent_differences)
    return basis
