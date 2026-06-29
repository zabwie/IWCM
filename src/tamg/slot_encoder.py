"""TAMGSlotEncoder — frame pairs → (N, 19) structured slots via spatial softmax.

Self-supervised: velocity MSE, contrastive identity, clustering type, reconstruction.

Layout: position(0-1) | velocity(2-3) | identity(4-11) | type(12-15) | misc(16-18)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# ── Channel layout for the 19-dim output ────────────────────────────────
POS = slice(0, 2)
VEL = slice(2, 4)
IDENTITY = slice(4, 12)
TYPE = slice(12, 16)
MISC = slice(16, 19)

# TAMG corrupt channels (same semantics, different indices vs oracle)
T_POS = slice(0, 2)
T_VEL = slice(2, 4)
T_TYPE = slice(12, 16)
T_ID = slice(4, 12)
T_EXIST = 15  # existence flag — reused from misc, old oracle compat


class TAMGSlotEncoder(nn.Module):
    """Raw frames → (N, 19) structured slots via spatial softmax + learned heads.

    Processes each frame independently with a shared CNN backbone, then uses
    learned slot queries to attend over spatial positions. Each query produces:
      - position (x, y) from attention-weighted spatial coordinates
      - slot feature vector from attention-weighted CNN features

    Per-slot heads then produce velocity, identity, and type from these features.
    """
    def __init__(self, num_slots=8, d_feat=64, img_size=64):
        super().__init__()
        self.num_slots = num_slots
        self.d_feat = d_feat

        # CNN backbone: 4 conv layers, stride 2 → 4×4 feature map at 64px
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1), nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2, 1), nn.ReLU(),
            nn.Conv2d(64, 64, 4, 2, 1), nn.ReLU(),
            nn.Conv2d(64, d_feat, 4, 2, 1), nn.ReLU(),
        )

        # Slot queries — each learns to track one object
        self.slot_queries = nn.Parameter(torch.randn(num_slots, d_feat) * 0.02)

        # Coordinate grid for spatial softmax (normalized to [0, 1])
        feat_size = img_size // 16  # 4 for 64px input
        coords = torch.linspace(0.5 / feat_size, 1.0 - 0.5 / feat_size, feat_size)
        gy, gx = torch.meshgrid(coords, coords, indexing='ij')
        pos_grid = torch.stack([gx.flatten(), gy.flatten()], dim=-1)  # (Np, 2)
        self.register_buffer('pos_grid', pos_grid)

        # ── Per-slot structured heads ───────────────────────────────────
        # Velocity: concat(slot_feat_t, slot_feat_t+1) → (vx, vy)
        # ponytail: tanh output keeps predictions bounded to [-1, 1], matching continuous-control displacement scale
        self.vel_head = nn.Sequential(
            nn.Linear(d_feat * 2, d_feat), nn.ReLU(),
            nn.Linear(d_feat, 2), nn.Tanh(),
        )
        # Identity: slot_feat → 8-dim embedding (contrastive)
        self.id_head = nn.Sequential(
            nn.Linear(d_feat, d_feat), nn.ReLU(),
            nn.Linear(d_feat, 8),
        )
        # Type: slot_feat → 4-dim soft cluster assignment
        self.type_head = nn.Sequential(
            nn.Linear(d_feat, d_feat), nn.ReLU(),
            nn.Linear(d_feat, 4),
        )
        # Misc projection — residual info the model needs
        self.misc_head = nn.Linear(d_feat, 3)

        # ── Lightweight decoder (reconstruction → prevent collapse) ─────
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(d_feat + 2, 64, 4, 2, 1), nn.ReLU(),   # 4→8
            nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.ReLU(),            # 8→16
            nn.ConvTranspose2d(32, 16, 4, 2, 1), nn.ReLU(),            # 16→32
            nn.ConvTranspose2d(16, 4, 4, 2, 1),                        # 32→64, +alpha
        )
        # Spatial anchoring — each slot starts biased to a grid cell
        grid_dim = math.ceil(math.sqrt(num_slots))
        cell_size = 1.0 / grid_dim
        anchors = torch.zeros(num_slots, 2)
        for k in range(num_slots):
            row, col = k // grid_dim, k % grid_dim
            anchors[k, 0] = (col + 0.5) * cell_size
            anchors[k, 1] = (row + 0.5) * cell_size
        self.anchor_positions = nn.Parameter(anchors)
        self.anchor_beta = 20.0
        # Temperature for slot-to-frame matching (vel loss)
        self.register_buffer('_match_temp', torch.tensor(10.0))

    # ── Frame encoding ──────────────────────────────────────────────────

    def encode_frame(self, img):
        """Encode one frame → positions, slot features, attention maps.

        Args:
            img: (B, 3, H, W) normalized [0,1] RGB frame.
        Returns:
            pos: (B, N, 2) normalized (x,y) per slot.
            feat: (B, N, d_feat) slot feature vectors.
            attn: (B, N, Np) attention weights over spatial positions.
        """
        B = img.shape[0]
        features = self.cnn(img)                           # (B, d_feat, Hf, Wf)
        C, Hf, Wf = features.shape[1], features.shape[2], features.shape[3]
        feat_flat = features.view(B, C, -1).transpose(1, 2)  # (B, Np, C)

        # Attention: each slot query attends over spatial positions
        attn_logits = torch.einsum('kc,bnc->bkn', self.slot_queries, feat_flat)

        # Spatial anchoring bias: slot k prefers positions near its anchor
        if hasattr(self, 'anchor_positions'):
            diff = self.anchor_positions.unsqueeze(1) - self.pos_grid.unsqueeze(0)
            dist_sq = (diff ** 2).sum(dim=-1)  # (N, Np)
            attn_logits = attn_logits - self.anchor_beta * dist_sq.unsqueeze(0)

        attn = F.softmax(attn_logits / math.sqrt(C), dim=-1)  # over positions

        # Position = weighted average of grid coordinates
        pos = torch.einsum('bkn,nc->bkc', attn, self.pos_grid)  # (B, N, 2)

        # Slot features = attention-weighted CNN features
        feat = torch.einsum('bkn,bnc->bkc', attn, feat_flat)   # (B, N, C)
        # ponytail: LayerNorm prevents unbounded feature growth during continuous-control training
        feat = F.layer_norm(feat, feat.shape[-1:])

        return pos, feat, attn

    # ── Slot matching across frames ─────────────────────────────────────

    def _match_slots(self, pos_t, pos_t1):
        """Greedy nearest-neighbor matching of slots between consecutive frames.

        Returns:
            perm: (B, N) index into pos_t1 giving best match for each slot in pos_t.
        """
        B, N, _ = pos_t.shape
        dist = torch.cdist(pos_t, pos_t1, p=2)  # (B, N, N)
        perm = torch.arange(N, device=pos_t.device).unsqueeze(0).expand(B, -1)
        matched_t = torch.zeros(B, N, dtype=torch.bool, device=pos_t.device)
        matched_t1 = torch.zeros(B, N, dtype=torch.bool, device=pos_t.device)

        for _ in range(N):
            d = dist.clone()
            # Mask rows and columns of already-matched slots
            row_mask = matched_t.unsqueeze(-1).expand(-1, -1, N)   # (B, N, N)
            col_mask = matched_t1.unsqueeze(1).expand(-1, N, -1)    # (B, N, N)
            d[row_mask | col_mask] = float('inf')

            if d.min() > 1.0:
                break
            # Per-batch: find best remaining match
            for b in range(B):
                flat_idx = d[b].argmin().item()
                s_t = flat_idx // N
                s_t1 = flat_idx % N
                if not matched_t[b, s_t] and not matched_t1[b, s_t1]:
                    perm[b, s_t] = s_t1
                    matched_t[b, s_t] = True
                    matched_t1[b, s_t1] = True
        return perm

    def _gather_matched(self, tensor, perm):
        """Gather slots in tensor according to matching permutation."""
        B, N, D = tensor.shape
        idx = perm.unsqueeze(-1).expand(-1, -1, D)
        return torch.gather(tensor, 1, idx)

    # ── Losses ───────────────────────────────────────────────────────────

    def _velocity_loss(self, pos_t, pos_t1, feat_t, feat_t1, perm):
        """MSE between predicted velocity and frame-difference velocity."""
        # Gather matched slots from t+1
        pos_t1_m = self._gather_matched(pos_t1, perm)
        feat_t1_m = self._gather_matched(feat_t1, perm)

        # True velocity = displacement in position space
        vel_true = pos_t1_m - pos_t  # (B, N, 2)
        # Clamp: grid-world objects rarely move >0.5/frame
        vel_true = torch.clamp(vel_true, -0.5, 0.5)

        # Predicted velocity from slot features
        vel_pred = self.vel_head(torch.cat([feat_t, feat_t1_m], dim=-1))

        return F.mse_loss(vel_pred, vel_true.detach())

    def _contrastive_loss(self, feat_t, feat_t1, perm, temp=0.1):
        """NT-Xent + uniformity: pushes same slot together, different slots apart."""
        id_t = self.id_head(feat_t)
        id_t1 = self.id_head(feat_t1)
        id_t1_m = self._gather_matched(id_t1, perm)
        id_t = F.normalize(id_t, dim=-1)
        id_t1_m = F.normalize(id_t1_m, dim=-1)
        B, N, D = id_t.shape

        # Raw cosine similarities (no temp) — for uniformity
        cos_sim = torch.einsum('bnd,bmd->bnm', id_t, id_t1_m)  # (B, N, N)

        # CE: positive pairs attract (with temperature)
        labels = torch.arange(N, device=id_t.device).unsqueeze(0).expand(B, -1)
        loss_ce = F.cross_entropy((cos_sim / temp).reshape(-1, N), labels.reshape(-1))

        # Uniformity: push different slots apart (within-frame, raw cosine)
        mask = ~torch.eye(N, dtype=torch.bool, device=id_t.device)
        loss_uniform = cos_sim[:, mask].pow(2).mean()

        return loss_ce + loss_uniform

    def _cluster_loss(self, feat_t):
        """Online clustering: slot features form distinct type clusters."""
        type_logits = self.type_head(feat_t)  # (B, N, 4)
        # Sinkhorn-Knopp: distribute slots evenly across types
        # (prevents all slots collapsing to one type)
        with torch.no_grad():
            soft = F.softmax(type_logits.detach(), dim=-1)  # (B, N, 4)
            # Sinkhorn normalization ~3 iterations
            for _ in range(3):
                soft = soft / soft.sum(dim=1, keepdim=True).clamp(min=1e-8)
                soft = soft / soft.sum(dim=2, keepdim=True).clamp(min=1e-8)
            targets = soft  # (B, N, 4) — pseudo-labels

        log_probs = F.log_softmax(type_logits, dim=-1)
        loss = -(targets * log_probs).sum(dim=-1).mean()
        return loss

    def _recon_loss(self, feat_t, img_t):
        """Decode slot features → frame MSE. Prevents collapse."""
        B, N, C = feat_t.shape
        Hf = Wf = 4
        # Position grid per slot: (B, N, Hf, Wf, 2)
        g = self.pos_grid.reshape(1, 1, Hf, Wf, 2)  # (1, 1, Hf, Wf, 2)
        pos = self.encode_frame(img_t)[0]  # (B, N, 2)
        pg = pos.reshape(B, N, 1, 1, 2).expand(-1, -1, Hf, Wf, 2)
        # Feature broadcast: (B, N, C, Hf, Wf)
        fb = feat_t.reshape(B, N, C, 1, 1).expand(-1, -1, -1, Hf, Wf)
        # Combine: (B, N, C+2, Hf, Wf)
        dec_in = torch.cat([fb, pg.permute(0, 1, 4, 2, 3)], dim=2)
        flat = dec_in.reshape(B * N, C + 2, Hf, Wf)
        raw = self.decoder(flat).reshape(B, N, 4, 64, 64)
        rgb, alpha = raw[:, :, :3], F.softmax(raw[:, :, 3:4], dim=1)
        return F.mse_loss((rgb * alpha).sum(dim=1), img_t)

    # ── Output projection ────────────────────────────────────────────────

    def _to_19(self, pos, feat, vel_pred=None):
        """Build (N, 19) output from internal representations."""
        id_emb = F.normalize(self.id_head(feat), dim=-1)
        type_soft = F.softmax(self.type_head(feat), dim=-1)
        misc = torch.tanh(self.misc_head(feat)) * 0.5 + 0.5

        if vel_pred is None:
            vel_pred = self.vel_head(torch.cat([feat, feat], dim=-1))

        # [pos(2) | vel(2) | identity(8) | type(4) | misc(3)] = 19
        slots = torch.cat([pos, vel_pred, id_emb, type_soft, misc], dim=-1)
        return slots

    def forward(self, Ft, Ft1):
        """Encode frame pair, compute losses, return slots + loss dict.

        Args:
            Ft: (B, 3, H, W) frame at time t.
            Ft1: (B, 3, H, W) frame at time t+1.
        Returns:
            slots_19: (B, N, 19) structured slot tensor.
            losses: dict of scalar losses for backprop.
        """
        B = Ft.shape[0]

        # Encode both frames
        pos_t, feat_t, _ = self.encode_frame(Ft)
        pos_t1, feat_t1, _ = self.encode_frame(Ft1)

        # Match slots between frames
        perm = self._match_slots(pos_t, pos_t1)

        # Compute losses
        loss_vel = self._velocity_loss(pos_t, pos_t1, feat_t, feat_t1, perm)
        loss_id = self._contrastive_loss(feat_t, feat_t1, perm)
        loss_type = self._cluster_loss(feat_t)
        loss_recon = self._recon_loss(feat_t, Ft)

        # Build 19-dim output (use matched features for velocity)
        feat_t1_m = self._gather_matched(feat_t1, perm)
        vel_pred = self.vel_head(torch.cat([feat_t, feat_t1_m], dim=-1))
        slots_19 = self._to_19(pos_t, feat_t, vel_pred)

        losses = {
            'loss_vel': loss_vel,
            'loss_id': loss_id,
            'loss_type': loss_type,
            'loss_recon': loss_recon,
        }
        return slots_19, losses


# ── Corruption function for TAMG slot layout ────────────────────────────

def corrupt_tamg_slots(Z, rng):
    """Apply compositional corruptions to TAMG-format (N, 19) slots."""
    is_numpy = isinstance(Z, np.ndarray)
    if is_numpy:
        Z = torch.from_numpy(Z)
    Zc = Z.clone()
    B, H, N, d = Z.shape
    T_POS = slice(0, 2)
    T_VEL = slice(2, 4)
    T_TYPE = slice(12, 16)
    T_ID = slice(4, 12)
    T_EXIST_single = 15

    for _ in range(rng.randint(2, 5)):
        op = rng.randint(0, 7)
        for i in range(B):
            ts = rng.randint(1, max(2, H - 2))
            s1 = rng.randint(0, N)
            s2 = rng.randint(0, N)
            while s2 == s1:
                s2 = rng.randint(0, N)
            e1 = Zc[i, ts, s1, T_EXIST_single].item() > 0.1 if T_EXIST_single < Z.shape[-1] else True
            e2 = Zc[i, ts, s2, T_EXIST_single].item() > 0.1 if T_EXIST_single < Z.shape[-1] else True

            if op == 0:  # Swap position+velocity between two slots
                # Swap position
                tmp = Zc[i, ts:, s1, T_POS].clone()
                Zc[i, ts:, s1, T_POS] = Zc[i, ts:, s2, T_POS].clone()
                Zc[i, ts:, s2, T_POS] = tmp
                tmp = Zc[i, ts:, s1, T_VEL].clone()
                Zc[i, ts:, s1, T_VEL] = Zc[i, ts:, s2, T_VEL].clone()
                Zc[i, ts:, s2, T_VEL] = tmp

            elif op == 1 and e1:  # Delete object
                pp = Zc[i, ts - 1, s1, T_POS].clone()
                Zc[i, ts:, s1, T_POS] = 0.0
                Zc[i, ts, s1, T_VEL] = -pp
                Zc[i, ts + 1:, s1, T_VEL] = 0.0
                if T_EXIST_single < Z.shape[-1]:
                    Zc[i, ts:, s1, T_EXIST_single] = 0.0

            elif op == 2 and e1:  # Add noise to position
                dx = (rng.random() - 0.5) * 0.5
                dy = (rng.random() - 0.5) * 0.5
                Zc[i, ts:, s1, 0] += dx
                Zc[i, ts:, s1, 1] += dy
                Zc[i, ts:, s1, :2].clamp_(0, 1)
                # Recompute velocity
                Zc[i, ts:, s1, T_VEL] = Zc[i, ts:, s1, T_POS] - Zc[i, ts-1:ts, s1, T_POS]

            elif op == 3 and e1 and not e2:  # Duplicate object
                Zc[i, :, s2] = Zc[i, :, s1].clone()
                Zc[i, :, s2, T_ID] = torch.rand(T_ID.stop - T_ID.start, device=Zc.device, dtype=Zc.dtype)
                Zc[i, :, s2, T_TYPE] = Zc[i, :, s1, T_TYPE].clone()
                if T_EXIST_single < Z.shape[-1]:
                    Zc[i, :, s2, T_EXIST_single] = 1.0

            elif op == 4:  # Reverse time segment
                sp = rng.randint(max(2, H // 3), max(3, 2 * H // 3))
                Zc[i, sp:] = Zc[i, sp:].flip(0)
                # Recompute velocity for reversed segment
                Zc[i, sp:, T_VEL] = Zc[i, sp:, T_POS] - Zc[i, sp-1:sp, T_POS]  # approximate

            elif op == 5 and e1:  # Change type
                nt = torch.zeros(4, device=Zc.device, dtype=Zc.dtype)
                nt[rng.randint(0, 4)] = 1.0
                Zc[i, ts:, s1, T_TYPE] = nt

            elif op == 6:  # Permute slots
                perm = list(range(N))
                rng.shuffle(perm)
                Zc[i] = Zc[i, :, perm]

    if is_numpy:
        return Zc.cpu().numpy()
    return Zc
