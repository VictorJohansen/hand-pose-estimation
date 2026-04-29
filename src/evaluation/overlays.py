"""Plotting utilities for hand keypoint prediction overlays.

Reusable across notebooks, evaluation scripts, and the live demo. Predicted
and ground-truth keypoints are drawn over an image; the hand skeleton is
optional. Colors and stroke weights are tunable.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib.figure
import matplotlib.pyplot as plt
import numpy as np

from src.data.freihand import HAND_CONNECTIONS


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FIGURES_DIR = PROJECT_ROOT / "reports" / "all-figures"

GT_COLOR = "#00ff66"
PRED_COLOR = "#ff3366"


def plot_keypoints(
    ax,
    image: np.ndarray,
    ground_truth: np.ndarray | None = None,
    predicted: np.ndarray | None = None,
    *,
    hand_connections: Sequence[tuple[int, int]] = HAND_CONNECTIONS,
    gt_color: str = GT_COLOR,
    pred_color: str = PRED_COLOR,
    linewidth: float = 1.0,
    marker_size: float = 12.0,
):
    """Draw an image with optional ground-truth and/or predicted keypoint overlays.

    Keypoint arrays must have shape (K, 2) with (x, y) pixel coordinates in the
    image. Either array may be omitted.
    """
    ax.imshow(image)
    for kp, color, label in (
        (ground_truth, gt_color, "ground truth"),
        (predicted, pred_color, "predicted"),
    ):
        if kp is None:
            continue
        kp = np.asarray(kp, dtype=np.float32)
        for start, end in hand_connections:
            ax.plot(
                [kp[start, 0], kp[end, 0]],
                [kp[start, 1], kp[end, 1]],
                color=color,
                linewidth=linewidth,
            )
        ax.scatter(kp[:, 0], kp[:, 1], s=marker_size, c=color, label=label)
    ax.axis("off")
    return ax


def prediction_grid(
    images: np.ndarray,
    ground_truth: np.ndarray,
    predicted: np.ndarray,
    *,
    titles: Sequence[str] | None = None,
    ncols: int = 2,
    figsize_per_panel: tuple[float, float] = (4.0, 4.0),
    suptitle: str | None = None,
    show_legend: bool = True,
) -> matplotlib.figure.Figure:
    """Draw a grid of prediction overlays for several samples.

    Inputs are batches: `images` (N, H, W, C), `ground_truth` (N, K, 2),
    `predicted` (N, K, 2). Returns the matplotlib Figure for further
    customization or saving via `save_figure()`.
    """
    n = len(images)
    if titles is None:
        titles = [None] * n
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(ncols * figsize_per_panel[0], nrows * figsize_per_panel[1]),
        squeeze=False,
    )
    flat_axes = list(axes.flat)
    for ax, img, gt, pred, title in zip(flat_axes, images, ground_truth, predicted, titles):
        plot_keypoints(ax, img, gt, pred)
        if title:
            ax.set_title(title)
    for ax in flat_axes[n:]:
        ax.axis("off")
    if show_legend and n > 0:
        flat_axes[0].legend(loc="upper right", fontsize=8)
    if suptitle:
        fig.suptitle(suptitle)
    fig.tight_layout()
    return fig


def save_figure(
    fig: matplotlib.figure.Figure,
    name: str,
    *,
    output_dir: Path = DEFAULT_FIGURES_DIR,
    dpi: int = 150,
) -> Path:
    """Save a figure to `reports/all-figures/<name>`. Defaults to `.png`."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    target = output_dir / name
    if target.suffix == "":
        target = target.with_suffix(".png")
    fig.savefig(target, dpi=dpi, bbox_inches="tight")
    return target


__all__ = [
    "DEFAULT_FIGURES_DIR",
    "GT_COLOR",
    "PRED_COLOR",
    "plot_keypoints",
    "prediction_grid",
    "save_figure",
]
