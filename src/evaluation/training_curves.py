"""Training curve plotting from saved Keras training histories.

Each run of `train_baseline` writes `logs/<run>/history.json`. These functions
load that file and turn it into a report-ready training/validation curve
figure without re-running training.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import matplotlib.figure
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LOGS_DIR = PROJECT_ROOT / "logs"

TRAIN_COLOR = "#3366cc"
VAL_COLOR = "#cc3366"


def load_history(
    run_name_or_path: str | Path,
    *,
    logs_dir: Path = DEFAULT_LOGS_DIR,
) -> dict[str, list[float]]:
    """Load a training history dict for the given run.

    Accepts either a run name (e.g. ``baseline_20260425_025144``), resolved to
    ``<logs_dir>/<run>/history.json``, or a direct path to a ``history.json``.
    """
    path = Path(run_name_or_path)
    target = path if path.is_file() else Path(logs_dir) / path.name / "history.json"
    return json.loads(target.read_text())


def plot_training_curves(
    history: dict[str, list[float]],
    *,
    metrics: Sequence[str] = ("loss", "mae"),
    suptitle: str | None = None,
    figsize_per_metric: tuple[float, float] = (5.0, 3.5),
) -> matplotlib.figure.Figure:
    """Plot train and validation curves for the requested metrics.

    ``history`` is the dict from ``history.json`` (or ``model.fit().history``).
    Each requested metric gets its own subplot, with ``<metric>`` (train) and
    ``val_<metric>`` (validation) overlaid. Missing metrics are silently
    skipped within the subplot.
    """
    n = len(metrics)
    fig, axes = plt.subplots(
        1,
        n,
        figsize=(n * figsize_per_metric[0], figsize_per_metric[1]),
        squeeze=False,
    )
    for ax, metric in zip(axes.flat, metrics):
        train = history.get(metric)
        val = history.get(f"val_{metric}")
        if train:
            ax.plot(range(1, 1 + len(train)), train, label="train", color=TRAIN_COLOR)
        if val:
            ax.plot(range(1, 1 + len(val)), val, label="val", color=VAL_COLOR)
        ax.set_xlabel("epoch")
        ax.set_ylabel(metric)
        ax.set_title(metric)
        ax.grid(True, alpha=0.3)
        if train or val:
            ax.legend()
    if suptitle:
        fig.suptitle(suptitle)
    fig.tight_layout()
    return fig


__all__ = [
    "DEFAULT_LOGS_DIR",
    "TRAIN_COLOR",
    "VAL_COLOR",
    "load_history",
    "plot_training_curves",
]
