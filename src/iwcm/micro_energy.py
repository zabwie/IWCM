"""MicroIWCM — latency-optimized IWCM for microsecond inference.

Optimization levels (cumulative):
  L0: Vanilla FusedIWCMEnergy (baseline: 137 μs/sample at B=1)
  L1: + Triton-fused temporal pooling (mean+max+var in 1 kernel)
  L2: + Triton-fused head MLP + aggregation (second kernel)
  L3: + CUDA Graph capture (zero kernel launch overhead)
  L4: + TF32 tensor cores + best-of-all

Goal: match or beat MLP throughput while preserving accuracy.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from .triton_ops import fused_temporal_pool, fused_head_forward


class MicroIWCM(nn.Module):
    """Latency-optimized IWCM energy function. Drop-in replacement for FusedIWCMEnergy.

    Same architecture, same weights — accelerated through:
      - Fused temporal pooling (Triton)
      - Fused head MLP (Triton)
      - CUDA Graph capture
      - TF32 tensor cores

    Args:
        d_slot: Slot feature dimension (19 for oracle slots).
        d_action: Action embedding dimension (11).
        hidden: Hidden dimension (128).
        num_slots: Max object slots (8).
    """

    def __init__(self, d_slot: int, d_action: int = 11, hidden: int = 128, num_slots: int = 8):
        super().__init__()
        self.hidden = hidden
        self.num_slots = num_slots

        self.shared = nn.Linear(d_slot, hidden)

        # Head MLP: same weights as FusedIWCMEnergy.head
        self.head = nn.Sequential(
            nn.Linear(hidden * 3, hidden),
            nn.GELU(),
            nn.Linear(hidden, 3),
        )
        self.register_buffer("lambdas", torch.tensor([1.0, 1.0, 1.5]))

        # CUDA Graph state
        self._graph: Optional[torch.cuda.CUDAGraph] = None
        self._graph_input_z0: Optional[torch.Tensor] = None
        self._graph_input_A: Optional[torch.Tensor] = None
        self._graph_input_Z: Optional[torch.Tensor] = None
        self._graph_output: Optional[torch.Tensor] = None
        self._static_pool = False
        self._use_triton_head = False

    # ------------------------------------------------------------------
    # Forward paths
    # ------------------------------------------------------------------

    def forward(self, z0: torch.Tensor, A: torch.Tensor, Z: torch.Tensor) -> torch.Tensor:
        """Compute energy. Dispatches to CUDA Graph if captured, else eager."""
        if self._graph is not None:
            return self._forward_graph(z0, A, Z)
        if self._use_triton_head:
            return self._forward_triton_full(z0, A, Z)
        if self._static_pool:
            return self._forward_triton_pool(z0, A, Z)
        return self._forward_eager(z0, A, Z)

    # --- L0: Vanilla eager (baseline, same as FusedIWCMEnergy) ---

    def _forward_eager(self, z0, A, Z):
        B, H, N, d = Z.shape
        Zf = self.shared(Z)
        Z_mean = Zf.mean(dim=1)
        Z_max = Zf.amax(dim=1)
        Z_sq = Zf * Zf
        Z_var = F.relu(Z_sq.mean(dim=1) - Z_mean * Z_mean)
        Z_std = torch.sqrt(Z_var + 1e-5)
        Zs = torch.cat([Z_mean, Z_max, Z_std], dim=-1)
        scores = self.head(Zs)
        agg = 0.3 * scores.mean(dim=1) + 0.7 * scores.amax(dim=1)
        return (agg * self.lambdas).sum(dim=-1)

    # --- L1: Triton temporal pool + native head ---

    def _forward_triton_pool(self, z0, A, Z):
        Zf = self.shared(Z)
        Z_mean, Z_max, Z_std = fused_temporal_pool(Zf, eps=1e-5)
        Zs = torch.cat([Z_mean, Z_max, Z_std], dim=-1)
        scores = self.head(Zs)
        agg = 0.3 * scores.mean(dim=1) + 0.7 * scores.amax(dim=1)
        return (agg * self.lambdas).sum(dim=-1)

    # --- L2: Triton temporal pool + Triton head ---

    def _forward_triton_full(self, z0, A, Z):
        Zf = self.shared(Z)
        Z_mean, Z_max, Z_std = fused_temporal_pool(Zf, eps=1e-5)
        stats = torch.cat([Z_mean, Z_max, Z_std], dim=-1)
        return fused_head_forward(
            stats,
            self.head[0].weight, self.head[0].bias,
            self.head[2].weight, self.head[2].bias,
            self.lambdas,
        )

    # --- L3: CUDA Graph ---

    def _forward_graph(self, z0, A, Z):
        self._graph_input_z0.copy_(z0)
        self._graph_input_A.copy_(A)
        self._graph_input_Z.copy_(Z)
        self._graph.replay()
        return self._graph_output.clone()

    def enable_triton_pool(self):
        """Use L1: fused temporal pooling via Triton (default head via PyTorch)."""
        self._static_pool = True
        self._use_triton_head = False
        self._graph = None

    def enable_triton_full(self):
        """Use L2: fused temporal pool + fused head (all Triton)."""
        self._static_pool = True
        self._use_triton_head = True
        self._graph = None

    def enable_cuda_graph(self, z0_sample, A_sample, Z_sample):
        """Capture CUDA Graph for zero-launch-overhead inference.

        Uses the Triton temporal pool + PyTorch head path (verified correct).
        After capture, all kernel launches (shared projection + temporal pool
        + head MLP + aggregation) are replayed in a single graph launch.

        Args:
            z0_sample, A_sample, Z_sample: Example inputs with the same shapes
                expected at inference time.
        """
        self._use_triton_head = False
        self._static_pool = True

        self._graph_input_z0 = z0_sample.clone()
        self._graph_input_A = A_sample.clone()
        self._graph_input_Z = Z_sample.clone()

        self._graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self._graph):
            self._graph_output = self._forward_triton_pool(
                self._graph_input_z0,
                self._graph_input_A,
                self._graph_input_Z,
            )

    def disable_graph(self):
        self._graph = None
        self._graph_input_z0 = None
        self._graph_input_A = None
        self._graph_input_Z = None
        self._graph_output = None

    def score_acceptance(self, z0, A, Z):
        return torch.sigmoid(-self.forward(z0, A, Z))


# ---------------------------------------------------------------------------
# Factory: create MicroIWCM from a trained FusedIWCMEnergy
# ---------------------------------------------------------------------------

def from_fused(fused_model: nn.Module) -> MicroIWCM:
    """Convert a trained FusedIWCMEnergy to MicroIWCM with weight transfer.

    Preserves all trained parameters.
    """
    micro = MicroIWCM(
        d_slot=fused_model.shared.in_features,
        d_action=11,
        hidden=fused_model.hidden,
        num_slots=8,
    )
    micro.load_state_dict(fused_model.state_dict())
    return micro


# ---------------------------------------------------------------------------
# NoPool IWCM — eliminates temporal reduction entirely
# ---------------------------------------------------------------------------

class NoPoolIWCM(nn.Module):
    """IWCM without temporal pooling — the head Linear absorbs the pooling.

    Instead of explicit mean/max/std reductions, the per-slot features
    are flattened across time and passed through a single wide Linear.
    The Linear(H*hidden, out) learns to approximate temporal statistics
    through its weights — no reduction kernels, no memory round-trips.

    Key insight: the Linear's matmul shape (B*N, H*hidden) @ (H*hidden, out)
    maps perfectly to GPU tensor cores (K=H*hidden=3200 is very wide),
    unlike the original head's (B*N, 384) @ (384, 128) which is too thin.

    Architecture:
      shared(Z): (B,H,N,d) → (B,H,N,h)     — same as FusedIWCMEnergy
      flatten:   (B,H,N,h) → (B,N,H*h)     — O(1) reshape, no compute
      head:      (B,N,H*h) → (B,N,out)     — single wide matmul
      agg:       mean+max over N → (B,out) — cheap, N=8
      lambda:    weighted sum → (B,)        — cheap, out=3

    Args:
        d_slot, d_action, hidden, num_slots: Same as FusedIWCMEnergy.
        head_hidden: If > 0, inserts a hidden layer (H*hidden→head_hidden→3)
                     for more capacity. Default 0 = direct H*hidden→3.
    """

    def __init__(self, d_slot: int, d_action: int = 11, hidden: int = 128,
                 num_slots: int = 8, head_hidden: int = 0):
        super().__init__()
        self.hidden = hidden
        self.num_slots = num_slots
        self.H = 25  # horizon

        self.shared = nn.Linear(d_slot, hidden)

        if head_hidden > 0:
            self.head = nn.Sequential(
                nn.Linear(self.H * hidden, head_hidden),
                nn.GELU(),
                nn.Linear(head_hidden, 3),
            )
        else:
            self.head = nn.Linear(self.H * hidden, 3)

        self.register_buffer("lambdas", torch.tensor([1.0, 1.0, 1.5]))

    def forward(self, z0, A, Z):
        B, H_in, N_in, d = Z.shape
        Zf = self.shared(Z)  # (B, H, N, hidden)

        # Flatten temporal: concatenate all timesteps per slot
        # (B, H, N, h) → permute → (B, N, H, h) → reshape → (B*N, H*h)
        Zf_flat = Zf.permute(0, 2, 1, 3).reshape(B * N_in, H_in * self.hidden)

        scores = self.head(Zf_flat).reshape(B, N_in, 3)  # (B, N, 3)

        # Aggregate across slots (N=8, trivially cheap)
        agg = 0.3 * scores.mean(dim=1) + 0.7 * scores.amax(dim=1)
        return (agg * self.lambdas).sum(dim=-1)

    def score_acceptance(self, z0, A, Z):
        return torch.sigmoid(-self.forward(z0, A, Z))
