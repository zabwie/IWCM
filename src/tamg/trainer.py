"""Self-Supervised AC3 Training Loop with TAMG and Validator Committee.

Algorithm 2 from Section 6.6:
  1. Encode trajectories → object-centric latents
  2. Apply TAMG operators → near-miss worlds
  3. Filter by manifold proximity + decode quality + validator disagreement
  4. Train W_θ: valid → low energy, hard → high energy, repair
  5. Train C_φ: reward acceptance + manifold + minimality + D + diversity - artifact

GPU-optimized: batched encoder + decoder inference, validators in parallel.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, List
from torch.utils.data import DataLoader
import numpy as np

from .corruptor import TAMGCorruptor
from .validators.committee import ValidatorCommittee
from .disagreement import DisagreementScorer
from ..iwcm.model import IWCM
from ..encoder.video_encoder import VideoEncoder
from ..encoder.decoder import VideoDecoder
from ..utils.logging import MetricsLogger


class TAMGTrainer:
    """TAMG self-supervised training loop.

    Trains IWCM world model, TAMG corruptor, and encoder/decoder
    without any symbolic oracle — using validator disagreement only.
    """

    def __init__(
        self,
        world_model: IWCM,
        corruptor: TAMGCorruptor,
        validators: ValidatorCommittee,
        encoder: VideoEncoder,
        decoder: VideoDecoder,
        logger: Optional[MetricsLogger] = None,
        device: str = "cuda",
        # Loss weights
        lambda_valid: float = 1.0,
        lambda_invalid: float = 1.0,
        lambda_repair: float = 0.5,
        lambda_disagreement: float = 1.0,
        lambda_manifold: float = 1.0,
        lambda_minimality: float = 0.3,
        lambda_diversity: float = 0.2,
        lambda_artifact: float = 0.5,
        # Curriculum
        disagreement_threshold: float = 0.3,
    ):
        self.world_model = world_model.to(device)
        self.corruptor = corruptor.to(device)
        self.validators = validators.to(device)
        self.encoder = encoder.to(device)
        self.decoder = decoder.to(device)
        self.logger = logger
        self.device = device

        # Optimizers
        self.opt_world = torch.optim.Adam(world_model.parameters(), lr=1e-4)
        self.opt_corruptor = torch.optim.Adam(corruptor.parameters(), lr=1e-4)
        self.opt_encoder = torch.optim.Adam(
            list(encoder.parameters()) + list(decoder.parameters()), lr=1e-4,
        )

        # Disagreement scorer
        self.d_scorer = DisagreementScorer(
            disagreement_threshold=disagreement_threshold,
        )

        # Loss weights
        self.lambda_valid = lambda_valid
        self.lambda_invalid = lambda_invalid
        self.lambda_repair = lambda_repair
        self.lambda_disagreement = lambda_disagreement
        self.lambda_manifold = lambda_manifold
        self.lambda_minimality = lambda_minimality
        self.lambda_diversity = lambda_diversity
        self.lambda_artifact = lambda_artifact

        self.global_step = 0

    def encode_video(self, video: torch.Tensor) -> torch.Tensor:
        """Encode video frames to latent worldline.

        Args:
            video: (B, H, C, W_img, H_img).

        Returns:
            Z: (B, H, N, d).
        """
        return self.encoder(video)

    def decode_slab(self, Z: torch.Tensor) -> torch.Tensor:
        """Decode worldline slab back to frames.

        Args:
            Z: (B, H, N, d).

        Returns:
            video: (B, H, C, W, H).
        """
        B, H, N, d = Z.shape
        recon = self.decoder(Z)  # (B, H, C, W, H)
        return recon

    def train_step(
        self, video: torch.Tensor, actions: torch.Tensor,
    ) -> dict:
        """Single TAMG training step.

        Args:
            video: (B, H, C, W, H) — rendered grid world frames.
            actions: (B, H, d_action) — action sequence.

        Returns:
            Dict of loss/metric values.
        """
        B, H, C, W_img, H_img = video.shape
        device = self.device

        # ─── Encode ───
        Z = self.encode_video(video)  # (B, H, N, d)
        z0 = Z[:, 0].mean(dim=1)  # (B, d) — initial state encoding
        A = actions  # (B, H, d_action)

        # ─── Generate corruptions ───
        Z_corrupted, corr_info = self.corruptor(Z, A)  # (B, H, N, d)

        # ─── Manifold check (decode quality) ───
        recon_orig = self.decode_slab(Z)
        recon_corr = self.decode_slab(Z_corrupted)
        losses = {}

        # Reconstruction loss (encoder training)
        loss_recon = F.mse_loss(recon_orig, video) + F.mse_loss(recon_corr, video) * 0.1
        losses["recon"] = loss_recon

        # Manifold loss: corrupted should still decode cleanly
        manifold_dist = F.mse_loss(recon_corr, video)
        losses["manifold"] = manifold_dist

        # ─── Validator committee ───
        valid_scores = self.validators(Z, z0, A)
        corr_scores = self.validators(Z_corrupted, z0, A)

        # Disagreement on corrupted
        d_score = self.d_scorer.compute_d_score(corr_scores)
        losses["disagreement"] = d_score.mean()

        # ─── World model training ───
        self.opt_world.zero_grad()

        energy_valid = self.world_model.energy(z0, A, Z)
        energy_corr = self.world_model.energy(z0, A, Z_corrupted)

        loss_valid = self.lambda_valid * energy_valid.mean()
        # Push corrupted energy high
        loss_invalid = self.lambda_invalid * F.relu(1.0 - energy_corr).mean()

        # Repair
        repaired_Z, repair_imp = self.world_model.repair(z0, A, Z_corrupted)
        loss_repair = self.lambda_repair * (-repair_imp.mean())

        loss_world = loss_valid + loss_invalid + loss_repair
        loss_world.backward()
        self.opt_world.step()

        # ─── Corruptor training ───
        self.validators.freeze_for_corruptor_update()

        self.opt_corruptor.zero_grad()

        # Re-evaluate after world model update
        with torch.no_grad():
            accept = self.world_model.score_accept(z0, A, Z_corrupted)
            # Minimality: ||Z' - Z||
            min_dist = (Z_corrupted - Z).pow(2).mean()

        # Corruptor reward (negative loss — maximize)
        loss_corr = -(
            accept.mean() +                           # W_θ acceptance
            self.lambda_disagreement * d_score.mean()  # validator disagreement
            - self.lambda_minimality * min_dist        # minimal edit
            - self.lambda_artifact * manifold_dist     # not an artifact
        )
        # Diversity bonus
        if "operator_alphas" in corr_info:
            alpha_diversity = corr_info["operator_alphas"].var(dim=0).mean()
            loss_corr -= self.lambda_diversity * alpha_diversity

        loss_corr.backward()
        self.opt_corruptor.step()

        self.validators.unfreeze()

        # ─── Encoder training ───
        self.opt_encoder.zero_grad()
        loss_recon.backward()
        self.opt_encoder.step()

        self.global_step += 1

        # ─── Metrics ───
        metrics = {
            "loss_world": loss_world.item(),
            "loss_corruptor": loss_corr.item(),
            "loss_recon": loss_recon.item(),
            "energy_valid": energy_valid.mean().item(),
            "energy_corr": energy_corr.mean().item(),
            "repair_improvement": repair_imp.mean().item(),
            "accept_mean": accept.mean().item(),
            "d_score_mean": d_score.mean().item(),
            "manifold_dist": manifold_dist.item(),
            "edit_distance": corr_info.get("edit_distance", 0.0),
        }

        if self.logger:
            self.logger.log_metrics(metrics, self.global_step)

        return metrics

    def train_epoch(self, dataloader: DataLoader) -> dict:
        """Train for one epoch.

        Args:
            dataloader: Yields (video, actions) batches.

        Returns:
            Averaged metrics over the epoch.
        """
        epoch_metrics = []

        for video, actions in dataloader:
            video = video.to(self.device)
            actions = actions.to(self.device)
            metrics = self.train_step(video, actions)
            epoch_metrics.append(metrics)

        avg = {}
        for key in epoch_metrics[0]:
            avg[key] = np.mean([m[key] for m in epoch_metrics])
        return avg
