"""Base PyTorch module with save/load utilities."""

from pathlib import Path
from typing import Optional
import torch
import torch.nn as nn


class BaseModel(nn.Module):
    """Base class for all IWCM models.

    Provides save/load/device utilities common to all model components.

    Usage:
        class MyModel(BaseModel):
            def __init__(self, config):
                super().__init__()
                self.config = config
                ...
    """

    def __init__(self):
        super().__init__()
        self._device = torch.device("cpu")

    def get_device(self) -> torch.device:
        """Return the device of the model's parameters.

        Returns:
            torch.device where model parameters reside (cpu if none).
        """
        try:
            return next(self.parameters()).device
        except StopIteration:
            return self._device

    def save(self, path: str) -> None:
        """Save model state dict and config to path.

        Args:
            path: File path for the checkpoint (directories created as needed).
        """
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        checkpoint = {
            "state_dict": self.state_dict(),
            "config": getattr(self, "config", None),
        }
        torch.save(checkpoint, path)

    def load(self, path: str, map_location: str = "cpu") -> None:
        """Load model state dict from path.

        Args:
            path: File path to checkpoint.
            map_location: Device to map tensors to.
        """
        checkpoint = torch.load(path, map_location=map_location, weights_only=True)
        # Only load state_dict; config is preserved from init
        self.load_state_dict(checkpoint["state_dict"])

    def count_parameters(self) -> int:
        """Return total number of trainable parameters.

        Returns:
            Integer count of parameters requiring gradients.
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def count_parameters_str(self) -> str:
        """Return human-readable parameter count.

        Returns:
            String like "1.2M" or "450K".
        """
        n = self.count_parameters()
        if n >= 1e6:
            return f"{n / 1e6:.1f}M"
        elif n >= 1e3:
            return f"{n / 1e3:.0f}K"
        return str(n)

    def to(self, *args, **kwargs):
        """Override to track device changes."""
        result = super().to(*args, **kwargs)
        if hasattr(result, "parameters"):
            try:
                self._device = next(result.parameters()).device
            except StopIteration:
                pass
        return result
