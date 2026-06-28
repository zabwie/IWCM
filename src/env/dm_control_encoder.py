"""DM Control oracle-structured slot encoder.

Maps DM Control physics states into IWCM-compatible oracle slot features,
analogous to the GridWorld oracle encoder. Each MuJoCo body becomes a slot
with channels for existence, type, position, velocity, orientation, and
identity hash — enabling the FusedIWCMEnergy to detect causal violations.

Design: 19-dim slots matching GRID_ORACLE_SLOT_DIM for compatibility with
the existing IWCM architecture that was validated on grid world data.

Channel layout (per slot):
  0:     existence flag (1.0 if slot has data, 0.0 if empty)
  1-4:   body type embedding (one-hot: 0=agent/root, 1=link, 2=joint, 3=end_effector)
  5-7:   normalized position (x, y, z)
  8-10:  normalized velocity (dx, dy, dz)
  11-14: orientation / joint angles (sin/cos pairs)
  15-18: identity hash (deterministic per body name)
"""

import numpy as np
from typing import Dict, List, Tuple, Optional

ORACLE_SLOT_DIM = 19
MAX_BODIES = 8


# ─── Domain Configurations ───────────────────────────────────────────────────

# Each config maps domain_name → list of body indices and their type,
# plus scale factors for normalization.
DOMAIN_CONFIGS: Dict[str, dict] = {
    'cartpole': {
        'bodies': [
            # Cart: qpos[0]=cart_x, qvel[0]=v_cart
            {'name': 'cart',    'body_idx': 1, 'type': 0,  'qpos_idx': [0],    'qvel_idx': [0]},
            # Pole: qpos[1]=theta (radians), qvel[1]=omega
            {'name': 'pole_1',  'body_idx': 2, 'type': 1,  'qpos_idx': [1],    'qvel_idx': [1]},
        ],
        'qpos_dim': 2,
        'qvel_dim': 2,
        'pos_scale': 5.0,
        'vel_scale': 3.0,
    },
    'cheetah': {
        'bodies': [
            {'name': 'torso',        'body_idx': 1,  'type': 0, 'qpos_idx': [0], 'qvel_idx': [0]},
            {'name': 'bfoot',        'body_idx': 2,  'type': 3, 'qpos_idx': [1], 'qvel_idx': [1]},
            {'name': 'bshin',        'body_idx': 3,  'type': 1, 'qpos_idx': [2], 'qvel_idx': [2]},
            {'name': 'bthigh',       'body_idx': 4,  'type': 1, 'qpos_idx': [3], 'qvel_idx': [3]},
            {'name': 'ffoot',        'body_idx': 5,  'type': 3, 'qpos_idx': [4], 'qvel_idx': [4]},
            {'name': 'fshin',        'body_idx': 6,  'type': 1, 'qpos_idx': [5], 'qvel_idx': [5]},
            {'name': 'fthigh',       'body_idx': 7,  'type': 1, 'qpos_idx': [6], 'qvel_idx': [6]},
        ],
        'state_dim': 17,
        'pos_scale': 15.0,
        'vel_scale': 10.0,
        'qpos_dim': 8,
        'qvel_dim': 7,
    },
    'walker': {
        'bodies': [
            {'name': 'torso',        'body_idx': 1,  'type': 0},
            {'name': 'foot',         'body_idx': 2,  'type': 3},
            {'name': 'foot_left',    'body_idx': 3,  'type': 3},
            {'name': 'leg',          'body_idx': 4,  'type': 1},
            {'name': 'leg_left',     'body_idx': 5,  'type': 1},
            {'name': 'thigh',        'body_idx': 6,  'type': 1},
            {'name': 'thigh_left',   'body_idx': 7,  'type': 1},
        ],
        'state_dim': 23,
        'pos_scale': 10.0,
        'vel_scale': 8.0,
    },
}


def _hash_name(name: str, n_channels: int = 4) -> np.ndarray:
    h = abs(hash(name)) % 1000000
    return np.array([
        ((h // 1) % 1000) / 1000.0,
        ((h // 1000) % 1000) / 1000.0,
        ((h // 1000000) % 1000) / 1000.0,
        ((h // 1000000000) % 1000) / 1000.0,
    ])[:n_channels]


class DMControlOracleEncoder:
    """Oracle-structured slot encoder for DM Control environments.

    Extracts per-body physics features (position, velocity, joint angles)
    and encodes them as IWCM-compatible oracle slots.

    Args:
        domain_name: DM Control domain (e.g., 'cartpole', 'cheetah').
        max_bodies: Maximum number of slots.
        slot_dim: Dimension per slot (must match IWCM d_slot).
    """

    def __init__(
        self,
        domain_name: str = 'cartpole',
        max_bodies: int = MAX_BODIES,
        slot_dim: int = ORACLE_SLOT_DIM,
    ):
        self.domain_name = domain_name
        self.max_bodies = max_bodies
        self.slot_dim = slot_dim

        if domain_name not in DOMAIN_CONFIGS:
            raise ValueError(
                f"Unknown domain '{domain_name}'. Available: {list(DOMAIN_CONFIGS.keys())}. "
                f"Add config to DOMAIN_CONFIGS or use a supported domain."
            )

        self.config = DOMAIN_CONFIGS[domain_name]
        self.bodies = self.config['bodies']
        self.pos_scale = self.config['pos_scale']
        self.vel_scale = self.config['vel_scale']

    def encode(
        self,
        qpos: np.ndarray,
        qvel: np.ndarray,
    ) -> np.ndarray:
        """Encode a single DM Control physics state into oracle slots.

        Uses raw MuJoCo qpos/qvel for linear position/velocity values
        (not cos/sin encoded observations which mask changes near vertical).

        Args:
            qpos: Generalized positions of shape (qpos_dim,).
            qvel: Generalized velocities of shape (qvel_dim,).

        Returns:
            Slots of shape (max_bodies, slot_dim).
        """
        slots = np.zeros((self.max_bodies, self.slot_dim), dtype=np.float32)

        for i, body_cfg in enumerate(self.bodies):
            if i >= self.max_bodies:
                break

            name = body_cfg['name']
            body_type = body_cfg['type']

            # Existence
            slots[i, 0] = 1.0

            # Type embedding (channels 1-4): one-hot per type
            slots[i, 1 + body_type] = 1.0

            # Position (channels 5-7): from qpos
            pidx_list = body_cfg.get('qpos_idx', [])
            for j, pidx in enumerate(pidx_list[:3]):
                if pidx < len(qpos):
                    slots[i, 5 + j] = qpos[pidx] / self.pos_scale

            # Velocity (channels 8-10): from qvel
            vidx_list = body_cfg.get('qvel_idx', [])
            for j, vidx in enumerate(vidx_list[:3]):
                if vidx < len(qvel):
                    slots[i, 8 + j] = qvel[vidx] / self.vel_scale

            # Orientation/extra (channels 11-14): raw qpos values unscaled
            # for linear sensitivity to changes
            for j, pidx in enumerate(pidx_list[:4]):
                if pidx < len(qpos):
                    slots[i, 11 + j] = qpos[pidx]

            # Identity hash (channels 15-18)
            h = _hash_name(name, 4)
            slots[i, 15:19] = h

        return slots

    def encode_trajectory(
        self,
        physics_states: list,
        actions: np.ndarray,
        horizon: int = 25,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Encode a full trajectory into (z0, A, Z) IWCM format.

        Args:
            physics_states: List of (qpos, qvel) tuples, length >= horizon+1.
            actions: Action array of shape (horizon, action_dim).
            horizon: Number of steps to encode.

        Returns:
            z0: (max_bodies, slot_dim) — initial oracle slots.
            A: (horizon, action_dim) — action sequence.
            Z: (horizon, max_bodies, slot_dim) — state slot sequence.
        """
        if len(physics_states) < horizon + 1:
            return None

        qpos0, qvel0 = physics_states[0]
        z0 = self.encode(qpos0, qvel0)
        A = actions[:horizon].astype(np.float32)

        Z_list = []
        for t in range(horizon):
            qpos, qvel = physics_states[t + 1]
            Z_list.append(self.encode(qpos, qvel))
        Z = np.stack(Z_list, axis=0)

        return z0, A, Z

    def encode_with_physics(
        self,
        physics,
        body_positions: Optional[np.ndarray] = None,
        body_velocities: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Encode using direct MuJoCo physics data for richer features.

        Args:
            physics: dm_control MuJoCo physics object.
            body_positions: Optional world positions (nbody, 3).
            body_velocities: Optional body velocities (nbody, 6).

        Returns:
            Slots of shape (max_bodies, slot_dim).
        """
        if body_positions is None:
            body_positions = physics.data.xpos.copy()
        if body_velocities is None:
            body_velocities = physics.data.cvel.copy()

        slots = np.zeros((self.max_bodies, self.slot_dim), dtype=np.float32)

        for i, body_cfg in enumerate(self.bodies):
            if i >= self.max_bodies:
                break

            name = body_cfg['name']
            body_type = body_cfg['type']
            bidx = body_cfg['body_idx']

            if bidx >= len(body_positions):
                continue

            # Existence
            slots[i, 0] = 1.0

            # Type
            slots[i, 1 + body_type] = 1.0

            # Position (world coordinates)
            pos = body_positions[bidx]
            slots[i, 5] = pos[0] / self.pos_scale
            slots[i, 6] = pos[1] / self.pos_scale
            if len(pos) > 2:
                slots[i, 7] = pos[2] / self.pos_scale

            # Velocity (linear)
            vel = body_velocities[bidx]
            slots[i, 8] = vel[0] / self.vel_scale
            slots[i, 9] = vel[1] / self.vel_scale
            if vel.shape[0] > 2:
                slots[i, 10] = vel[2] / self.vel_scale

            # Identity hash
            h = _hash_name(name, 4)
            slots[i, 15:19] = h

        return slots


def encode_dm_control_state(
    domain_name: str,
    state: np.ndarray,
    max_bodies: int = MAX_BODIES,
) -> np.ndarray:
    """Convenience: encode a single state into oracle slots.

    Args:
        domain_name: DM Control domain name.
        state: Flat state vector.
        max_bodies: Maximum number of slots.

    Returns:
        Oracle slots of shape (max_bodies, ORACLE_SLOT_DIM).
    """
    encoder = DMControlOracleEncoder(domain_name, max_bodies=max_bodies)
    return encoder.encode(state)
