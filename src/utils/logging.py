"""Dual logging: TensorBoard + Wandb with graceful fallback."""

from pathlib import Path
from typing import Optional, Dict, Any
import numpy as np


class MetricsLogger:
    """Unified logger wrapping TensorBoard and/or Wandb.

    Usage:
        logger = MetricsLogger(log_dir="outputs/logs", use_tensorboard=True)
        logger.log_metrics({"loss": 0.5, "acc": 0.8}, step=100)
        logger.close()
    """

    def __init__(
        self,
        log_dir: str = "outputs/logs",
        project_name: str = "iwcm",
        run_name: Optional[str] = None,
        use_wandb: bool = False,
        use_tensorboard: bool = False,
    ):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._tb_writer = None
        self._wandb_run = None
        self.use_wandb = use_wandb
        self.use_tensorboard = use_tensorboard

        if use_tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter
                self._tb_writer = SummaryWriter(log_dir=str(self.log_dir / "tensorboard"))
            except ImportError:
                print("[MetricsLogger] tensorboard not installed; skipping.")

        if use_wandb:
            try:
                import wandb
                self._wandb_run = wandb.init(
                    project=project_name,
                    name=run_name,
                    dir=str(self.log_dir / "wandb"),
                )
            except ImportError:
                print("[MetricsLogger] wandb not installed; skipping.")
            except Exception as e:
                print(f"[MetricsLogger] wandb init failed: {e}")

    def log_metrics(self, metrics: Dict[str, Any], step: int) -> None:
        """Log scalar metrics to all active backends.

        Args:
            metrics: Dict mapping metric names to scalar values.
            step: Global step for x-axis.
        """
        if self._tb_writer:
            for key, value in metrics.items():
                if isinstance(value, (int, float, np.floating, np.integer)):
                    self._tb_writer.add_scalar(key, float(value), step)
        if self._wandb_run:
            try:
                import wandb
                wandb.log({k: float(v) if hasattr(v, "__float__") else v
                          for k, v in metrics.items()}, step=step)
            except Exception:
                pass

    def log_worldline_viz(self, fig: Any, tag: str, step: int) -> None:
        """Log a matplotlib figure as an image.

        Args:
            fig: Matplotlib Figure object.
            tag: Name/label for the visualization.
            step: Global step.
        """
        if self._tb_writer:
            self._tb_writer.add_figure(tag, fig, step)
        if self._wandb_run:
            try:
                import wandb
                wandb.log({tag: wandb.Image(fig)}, step=step)
            except Exception:
                pass

    def log_histogram(self, values: Any, tag: str, step: int) -> None:
        """Log histogram of tensor values.

        Args:
            values: Tensor or numpy array of values.
            tag: Name/label.
            step: Global step.
        """
        if hasattr(values, "detach"):
            values = values.detach().cpu().numpy()
        if isinstance(values, np.ndarray):
            values = values.flatten()

        if self._tb_writer:
            self._tb_writer.add_histogram(tag, values, step)
        if self._wandb_run:
            try:
                import wandb
                wandb.log({tag: wandb.Histogram(values)}, step=step)
            except Exception:
                pass

    def log_text(self, text: str, tag: str, step: int) -> None:
        """Log arbitrary text.

        Args:
            text: String content.
            tag: Name/label.
            step: Global step.
        """
        if self._tb_writer:
            self._tb_writer.add_text(tag, text, step)
        if self._wandb_run:
            try:
                import wandb
                wandb.log({tag: text}, step=step)
            except Exception:
                pass

    def close(self) -> None:
        """Close all logging backends."""
        if self._tb_writer:
            self._tb_writer.close()
        if self._wandb_run:
            try:
                self._wandb_run.finish()
            except Exception:
                pass
