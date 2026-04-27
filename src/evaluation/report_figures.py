"""Generate report-ready figures for the trained hand pose runs.

The figures are built from the canonical validation split, saved run histories,
and saved evaluation summaries. Prediction overlays additionally require the
local checkpoints in ``models/<run>/best.keras``.

Run from the project root:
    python -m src.evaluation.report_figures
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Sequence

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import keras
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from src.data.freihand import FreiHand, SPLIT_SEED, SPLIT_VALIDATION_FRACTION
from src.evaluation.metrics import sample_mpke
from src.evaluation.overlays import DEFAULT_FIGURES_DIR, plot_keypoints


PROJECT_ROOT = Path(__file__).resolve().parents[2]
LOGS_DIR = PROJECT_ROOT / "logs"
MODELS_DIR = PROJECT_ROOT / "models"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"

DEFAULT_RUNS = ("baseline-model-1", "baseline-model-2", "improved-model-1")
DEFAULT_DATASET_SAMPLES = (15550, 28457, 18199, 6097)

RUN_LABELS = {
    "baseline-model-1": "Baseline 1",
    "baseline-model-2": "Baseline 2",
    "improved-model-1": "Improved 1",
}

RUN_COLORS = {
    "baseline-model-1": "#4c78a8",
    "baseline-model-2": "#f58518",
    "improved-model-1": "#54a24b",
}

SUMMARY_METRICS = (
    ("median_sample_mpke_px", "Median"),
    ("p75_sample_mpke_px", "p75"),
    ("p90_sample_mpke_px", "p90"),
    ("p95_sample_mpke_px", "p95"),
)


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def _run_label(run_name: str) -> str:
    return RUN_LABELS.get(run_name, run_name)


def _run_colors(run_names: Sequence[str]) -> list[str]:
    return [RUN_COLORS.get(run_name, "#767676") for run_name in run_names]


def _resolve_input_size(config: dict) -> int:
    input_shape = config.get("input_shape")
    if input_shape:
        return int(input_shape[0])
    return 224


def _load_run_summary(run_name: str) -> dict:
    config_path = LOGS_DIR / run_name / "config.json"
    evaluation_path = ARTIFACTS_DIR / run_name / "evaluation.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing config: {config_path}")
    if not evaluation_path.exists():
        raise FileNotFoundError(
            f"Missing evaluation: {evaluation_path}. "
            f"Run `python -m src.evaluation.evaluate_run {run_name}` first."
        )

    config = _read_json(config_path)
    evaluation = _read_json(evaluation_path)
    return {
        "run_name": run_name,
        "label": _run_label(run_name),
        "config": config,
        "evaluation": evaluation,
        "metrics": evaluation["metrics"],
        "param_count": int(evaluation["param_count"]),
        "input_size": int(evaluation.get("input_size") or _resolve_input_size(config)),
    }


def _load_history(run_name: str) -> dict[str, list[float]]:
    history_path = LOGS_DIR / run_name / "history.json"
    if not history_path.exists():
        raise FileNotFoundError(f"Missing history: {history_path}")
    return _read_json(history_path)


def _is_heatmap_output(output_shape) -> bool:
    return len(output_shape) == 4


def _load_keypoint_model(run_name: str, input_size: int) -> keras.Model:
    checkpoint = MODELS_DIR / run_name / "best.keras"
    if not checkpoint.exists():
        raise FileNotFoundError(f"Missing checkpoint: {checkpoint}")

    model = keras.models.load_model(str(checkpoint))
    if _is_heatmap_output(model.output_shape):
        from src.models.heatmaps import wrap_with_keypoint_decoder

        return wrap_with_keypoint_decoder(model, input_size=input_size)
    return model


def _normalize_keypoints(array: np.ndarray) -> np.ndarray:
    array = np.asarray(array, dtype=np.float32)
    if array.ndim == 2:
        array = array.reshape(array.shape[0], 21, 2)
    return array


def _save_figure(
    fig,
    stem: str,
    *,
    output_dir: Path,
    formats: Sequence[str] = ("png", "pdf"),
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for suffix in formats:
        path = output_dir / f"{stem}.{suffix}"
        fig.savefig(path, dpi=300, bbox_inches="tight")
        paths.append(path)
    plt.close(fig)
    return paths


def _apply_report_style() -> None:
    plt.rcParams.update(
        {
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.alpha": 0.25,
            "font.size": 10,
            "figure.titlesize": 12,
        }
    )


def plot_primary_results(summaries: Sequence[dict]):
    run_names = [summary["run_name"] for summary in summaries]
    labels = [summary["label"] for summary in summaries]
    colors = _run_colors(run_names)
    mpke_values = [summary["metrics"]["mpke_px"] for summary in summaries]
    param_values = [summary["param_count"] / 1_000_000 for summary in summaries]

    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.4))

    axes[0].bar(labels, mpke_values, color=colors)
    axes[0].set_title("Validation MPKE")
    axes[0].set_ylabel("pixels, lower is better")
    axes[0].set_ylim(0, max(mpke_values) * 1.22)
    for index, value in enumerate(mpke_values):
        axes[0].text(index, value + 0.35, f"{value:.1f}", ha="center", va="bottom")

    axes[1].bar(labels, param_values, color=colors)
    axes[1].set_title("Model size")
    axes[1].set_ylabel("parameters, millions")
    axes[1].set_yscale("log")
    axes[1].set_ylim(0.08, max(param_values) * 2.2)
    for index, value in enumerate(param_values):
        axes[1].text(index, value * 1.12, f"{value:.2f}M", ha="center", va="bottom")

    for ax in axes:
        ax.tick_params(axis="x", rotation=15)
    fig.tight_layout()
    return fig


def plot_error_summary(summaries: Sequence[dict]):
    labels = [summary["label"] for summary in summaries]
    x = np.arange(len(summaries), dtype=np.float32)
    width = 0.18
    offsets = (np.arange(len(SUMMARY_METRICS)) - (len(SUMMARY_METRICS) - 1) / 2) * width
    colors = ("#4c78a8", "#72b7b2", "#f58518", "#e45756")

    fig, ax = plt.subplots(figsize=(8.2, 3.6))
    for offset, (metric_key, metric_label), color in zip(offsets, SUMMARY_METRICS, colors):
        values = [summary["metrics"][metric_key] for summary in summaries]
        ax.bar(x + offset, values, width=width, label=metric_label, color=color)

    ax.set_title("Per-sample MPKE distribution")
    ax.set_ylabel("pixels, lower is better")
    ax.set_xticks(x, labels)
    ax.tick_params(axis="x", rotation=15)
    ax.legend(ncols=len(SUMMARY_METRICS), frameon=False, loc="upper left")
    fig.tight_layout()
    return fig


def plot_training_curves(run_names: Sequence[str]):
    fig, axes = plt.subplots(len(run_names), 2, figsize=(8.8, 2.45 * len(run_names)), squeeze=False)
    for row, run_name in enumerate(run_names):
        history = _load_history(run_name)
        for col, metric in enumerate(("loss", "mae")):
            ax = axes[row][col]
            train = history.get(metric, [])
            val = history.get(f"val_{metric}", [])
            if train:
                ax.plot(range(1, len(train) + 1), train, label="train", color="#4c78a8", linewidth=1.8)
            if val:
                ax.plot(range(1, len(val) + 1), val, label="val", color="#e45756", linewidth=1.8)
            ax.set_title(f"{_run_label(run_name)}: {metric}")
            ax.set_xlabel("epoch")
            ax.set_ylabel(metric)
            ax.legend(frameon=False, loc="best")
    fig.tight_layout()
    return fig


def plot_dataset_examples(sample_ids: Sequence[int], *, input_size: int = 224):
    dataset = FreiHand()
    dataset.validate()
    images, keypoints = dataset.load_batch(
        sample_ids,
        image_size=(input_size, input_size),
        normalize_images=True,
        flatten_keypoints=False,
    )

    fig, axes = plt.subplots(1, len(sample_ids), figsize=(3.1 * len(sample_ids), 3.05), squeeze=False)
    for ax, image, keypoint, sample_id in zip(axes.flat, images, keypoints, sample_ids):
        plot_keypoints(
            ax,
            image,
            ground_truth=keypoint,
            predicted=None,
            linewidth=1.25,
            marker_size=9,
        )
        ax.set_title(f"sample {int(sample_id)}")
    fig.tight_layout()
    return fig


def _validation_indices(dataset: FreiHand) -> np.ndarray:
    _, val_idx = dataset.train_validation_split(
        validation_fraction=SPLIT_VALIDATION_FRACTION,
        seed=SPLIT_SEED,
    )
    return val_idx


def _representative_positions(summary: dict) -> list[tuple[str, int]]:
    representative = summary["metrics"].get("representative_indices")
    if not representative:
        raise ValueError(
            f"Run {summary['run_name']} does not include representative indices. "
            f"Run `python -m src.evaluation.evaluate_run {summary['run_name']}` first."
        )
    return [
        ("easy", int(representative["best"])),
        ("typical", int(representative["median"])),
        ("hard", int(representative["p90"])),
        ("worst", int(representative["worst"])),
    ]


def plot_prediction_comparison(summaries: Sequence[dict], *, reference_run: str):
    summary_by_run = {summary["run_name"]: summary for summary in summaries}
    if reference_run not in summary_by_run:
        raise ValueError(f"Reference run '{reference_run}' was not included.")

    dataset = FreiHand()
    dataset.validate()
    val_idx = _validation_indices(dataset)
    examples = _representative_positions(summary_by_run[reference_run])
    example_positions = [position for _, position in examples]
    sample_ids = val_idx[example_positions]

    nrows = len(summaries)
    ncols = len(examples)
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.1 * ncols, 2.85 * nrows), squeeze=False)

    for row, summary in enumerate(summaries):
        run_name = summary["run_name"]
        input_size = summary["input_size"]
        model = _load_keypoint_model(run_name, input_size)
        images, targets = dataset.load_batch(
            sample_ids,
            image_size=(input_size, input_size),
            normalize_images=True,
            flatten_keypoints=False,
        )
        predictions = _normalize_keypoints(model.predict(images, verbose=0))
        targets = _normalize_keypoints(targets)
        errors = sample_mpke(predictions, targets)

        for col, ((difficulty, _), sample_id) in enumerate(zip(examples, sample_ids)):
            ax = axes[row][col]
            plot_keypoints(
                ax,
                images[col],
                targets[col],
                predictions[col],
                linewidth=1.25,
                marker_size=9,
            )
            if row == 0:
                ax.set_title(f"{difficulty}\nsample {int(sample_id)}")
            ax.text(
                0.02,
                0.98,
                f"{_run_label(run_name)}\n{errors[col]:.1f} px",
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=8,
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.72, "pad": 2.5},
            )
            if row == 0 and col == 0:
                ax.legend(loc="lower right", fontsize=7, framealpha=0.78)

    fig.tight_layout()
    return fig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate report-ready figures from saved hand pose runs.",
    )
    parser.add_argument(
        "runs",
        nargs="*",
        default=list(DEFAULT_RUNS),
        help="Run names to include. Defaults to all recorded project runs.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_FIGURES_DIR,
        help="Directory where report figures are written.",
    )
    parser.add_argument(
        "--reference-run",
        default="baseline-model-1",
        help="Run whose representative validation samples define the overlay grid.",
    )
    parser.add_argument(
        "--dataset-samples",
        type=int,
        nargs="+",
        default=list(DEFAULT_DATASET_SAMPLES),
        help="FreiHAND training sample IDs to show in the dataset example figure.",
    )
    parser.add_argument(
        "--skip-predictions",
        action="store_true",
        help="Skip the prediction overlay figure, which requires local checkpoints and data.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _apply_report_style()

    summaries = [_load_run_summary(run_name) for run_name in args.runs]
    generated: list[Path] = []

    generated += _save_figure(
        plot_dataset_examples(args.dataset_samples),
        "report_dataset_examples",
        output_dir=args.output_dir,
    )
    generated += _save_figure(
        plot_primary_results(summaries),
        "report_primary_results",
        output_dir=args.output_dir,
    )
    generated += _save_figure(
        plot_error_summary(summaries),
        "report_error_summary",
        output_dir=args.output_dir,
    )
    generated += _save_figure(
        plot_training_curves(args.runs),
        "report_training_curves",
        output_dir=args.output_dir,
    )

    if not args.skip_predictions:
        generated += _save_figure(
            plot_prediction_comparison(summaries, reference_run=args.reference_run),
            "report_prediction_comparison",
            output_dir=args.output_dir,
        )

    for path in generated:
        print(path.relative_to(PROJECT_ROOT))


if __name__ == "__main__":
    main()
