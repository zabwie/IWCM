"""DM Control environment wrapper for IWCM world model pipeline.

Provides a consistent interface wrapping dm_control environments,
producing state observations, rendered frames, and action sequences
compatible with the IWCM training loop.

Architecture:
  DMControlWrapper
    └── dm_control environment (MuJoCo physics)
    └── generates trajectories: (frames, state_dict, actions)
    └── optional pixel rendering at configurable resolution

Usage:
  wrapper = DMControlWrapper('cartpole', 'swingup', seed=42)
  frames, states, actions = wrapper.generate_trajectory(horizon=25)
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dm_control import suite
from dm_control.suite import common


class DMControlWrapper:
    """Wrapper around dm_control suite environments.

    Args:
        domain_name: DM Control domain (e.g., 'cartpole', 'cheetah', 'walker').
        task_name: Task within domain (e.g., 'swingup', 'run', 'walk').
        seed: Random seed for reproducibility.
        render_size: Pixel resolution for rendered frames (height, width).
        action_repeat: Number of times to repeat each action.
        max_episode_steps: Maximum steps before truncation.
    """

    def __init__(
        self,
        domain_name: str = 'cartpole',
        task_name: str = 'swingup',
        seed: int = 42,
        render_size: Tuple[int, int] = (64, 64),
        action_repeat: int = 1,
        max_episode_steps: int = 250,
    ):
        self.domain_name = domain_name
        self.task_name = task_name
        self.seed = seed
        self.render_size = render_size
        self.action_repeat = action_repeat
        self.max_episode_steps = max_episode_steps

        self._env = suite.load(domain_name, task_name, task_kwargs={'random': seed})
        self._rng = np.random.RandomState(seed)

        # Cache state/action dimensions
        ts = self._env.reset()
        self.state_dim = sum(
            int(np.prod(v.shape)) if hasattr(v, 'shape') else 1
            for v in ts.observation.values()
        )
        self.action_spec = self._env.action_spec()
        self.action_dim = int(np.prod(self.action_spec.shape))
        self.action_min = self.action_spec.minimum
        self.action_max = self.action_spec.maximum

    def reset(self) -> Dict[str, np.ndarray]:
        ts = self._env.reset()
        return dict(ts.observation)

    def step(self, action: np.ndarray) -> Dict[str, np.ndarray]:
        ts = self._env.step(action)
        return dict(ts.observation)

    def render(self) -> np.ndarray:
        h, w = self.render_size
        return self._env.physics.render(camera_id=0, height=h, width=w)

    def sample_action(self) -> np.ndarray:
        return self._rng.uniform(
            self.action_min, self.action_max, size=self.action_spec.shape
        )

    def generate_trajectory(
        self,
        horizon: int = 25,
        random_policy: bool = True,
    ) -> Optional[Tuple[np.ndarray, List[Dict], np.ndarray]]:
        """Generate a trajectory of given horizon.

        Args:
            horizon: Number of steps to collect.
            random_policy: Use random actions if True.

        Returns:
            frames: (horizon+1, H, W, 3) rendered frames (grabs initial frame too).
            states: list of horizon+1 state dicts.
            actions: (horizon,) action array.
            None if episode ends before horizon steps.
        """
        ts = self._env.reset()
        states = [dict(ts.observation)]
        physics_states = [(self._env.physics.data.qpos.copy(),
                           self._env.physics.data.qvel.copy())]
        actions_list = []

        for _ in range(horizon):
            if random_policy:
                action = self.sample_action()
            else:
                action = np.zeros(self.action_spec.shape)

            ts = self._env.step(action)
            states.append(dict(ts.observation))
            physics_states.append((self._env.physics.data.qpos.copy(),
                                   self._env.physics.data.qvel.copy()))
            actions_list.append(action.copy())

            if ts.last():
                return None

        actions = np.stack(actions_list, axis=0)  # (H, action_dim)
        return physics_states, states, actions

    def generate_corrupted_trajectory(
        self,
        horizon: int = 25,
        corruption_type: str = 'teleport',
        corruption_step: Optional[int] = None,
        rng: Optional[np.random.RandomState] = None,
    ) -> Optional[Tuple[np.ndarray, List[Dict], np.ndarray, Dict]]:
        """Generate a trajectory with a causal corruption injected.

        Corruption types:
          'teleport': Randomly perturb position/velocity at corruption_step.
          'freeze': Set velocity to zero from corruption_step onward.
          'reverse': Flip velocity sign at corruption_step.

        Args:
            horizon: Trajectory length.
            corruption_type: Type of corruption to inject.
            corruption_step: Step at which corruption occurs (None = random).
            rng: Random state for corruption parameters.

        Returns:
            frames, states, actions, meta dict.
        """
        if rng is None:
            rng = self._rng

        if corruption_step is None:
            corruption_step = rng.randint(horizon // 4, 3 * horizon // 4)

        ts = self._env.reset()
        states = [dict(ts.observation)]
        actions_list = []
        physics_states = [(self._env.physics.data.qpos.copy(),
                           self._env.physics.data.qvel.copy())]

        for t in range(horizon):
            action = self.sample_action()
            ts = self._env.step(action)
            obs = dict(ts.observation)
            states.append(obs)
            actions_list.append(action.copy())

            qpos = self._env.physics.data.qpos.copy()
            qvel = self._env.physics.data.qvel.copy()
            physics_states.append((qpos, qvel))

            if t == corruption_step:
                if corruption_type == 'teleport':
                    # ponytail: raised clamp from 0.5→2.0 so cheetah-scale states get detectable perturbations
                    qpos_std = np.clip(np.abs(qpos) * 0.2, 0.02, 2.0)
                    qvel_std = np.clip(np.abs(qvel) * 0.2, 0.1, 4.0)
                    self._env.physics.data.qpos += rng.randn(*qpos.shape) * qpos_std
                    self._env.physics.data.qvel += rng.randn(*qvel.shape) * qvel_std
                elif corruption_type == 'freeze':
                    # Set velocity to large random values (more detectable via max pooling)
                    qvel_mag = np.clip(np.abs(self._env.physics.data.qvel) * 2.0, 1.0, 5.0)
                    self._env.physics.data.qvel[:] = rng.randn(*qvel.shape) * qvel_mag
                elif corruption_type == 'reverse':
                    # Full velocity reversal with amplification
                    self._env.physics.data.qvel[:] *= -2.0

            if ts.last() and t < horizon - 1:
                return None

        actions = np.stack(actions_list, axis=0)
        meta = {
            'law_type': 'identity' if corruption_type in ['teleport'] else 'conservation',
            'violation_type': corruption_type,
            'corruption_step': corruption_step,
        }
        return physics_states, states, actions, meta

    def close(self):
        pass  # dm_control envs don't need explicit closing


def get_available_domains() -> List[str]:
    return sorted(set(
        t.split('_')[0] for t in suite.ALL_TASKS
        if not t.startswith('manipulator')  # requires robot model
    ))
