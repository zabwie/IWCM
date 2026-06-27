"""Reproducibility utilities."""

import random
import os
from typing import Optional

import numpy as np


def set_seed(seed: int = 42) -> None:
    """Set all random seeds for full reproducibility.

    Controls Python random, NumPy, PyTorch (if available),
    and CUDA determinism settings.

    Args:
        seed: Integer seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def seed_worker(worker_id: int) -> None:
    """DataLoader worker init function for reproducibility.

    Each worker gets a unique but deterministic random state based
    on the base seed and worker ID.

    Args:
        worker_id: DataLoader worker identifier.
    """
    try:
        import torch
        worker_seed = torch.initial_seed() % (2 ** 32)
        np.random.seed(worker_seed)
        random.seed(worker_seed)
    except ImportError:
        pass
