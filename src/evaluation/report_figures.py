"""Generate numbered PNG figures for the hand pose project report.

Run from the project root:
    python -m src.evaluation.report_figures baseline-model-1 improved-model-1 improved-model-1-online-augmented
"""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import keras
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from src.data.augmentations import augment_image_and_keypoints
from src.data.freihand import (
    HAND_CONNECTIONS,
    SPLIT_SEED,
    SPLIT_VALIDATION_FRACTION,
    VARIANTS,
    FreiHand,
    VariantName,
)
from src.evaluation.metrics import sample_mpke
from src.evaluation.overlays import GT_COLOR, PRED_COLOR, plot_keypoints


PROJECT_ROOT = Path(__file__).resolve().parents[2]
LOGS_DIR = PROJECT_ROOT / "logs"
MODELS_DIR = PROJECT_ROOT / "models"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
DEFAULT_REPORT_FIGURES_DIR = PROJECT_ROOT / "reports" / "report-figures"
DEFAULT_CAPTION_OUTPUT = PROJECT_ROOT / "docs" / "report-figure-captions.md"

DEFAULT_RUNS = (
    "baseline-model-1",
    "improved-model-1",
    "improved-model-1-online-augmented",
)
ARCHITECTURE_RUNS = ("baseline-model-1", "improved-model-1")
QUALITATIVE_RUNS = ("baseline-model-1", "improved-model-1")
DEFAULT_IMAGE_SIZE = 224
DEFAULT_DATASET_SAMPLE_IDS = (0, 1, 2, 3)

RUN_LABELS = {
    "baseline-model-1": "baseline",
    "baseline-model-2": "baseline 2",
    "improved-model-1": "improved",
    "improved-model-1-online-augmented": "webcam",
}

RUN_COLORS = {
    "baseline-model-1": "#3b6ea8",
    "baseline-model-2": "#9b6a2f",
    "improved-model-1": "#2f8f5b",
    "improved-model-1-online-augmented": "#7b5aa6",
}

SUMMARY_METRICS = (
    ("median_sample_mpke_px", "median"),
    ("p90_sample_mpke_px", "p90"),
    ("p95_sample_mpke_px", "p95"),
)


@dataclass(frozen=True)
class FigureContext:
    run_names: tuple[str, ...]
    summaries: tuple[dict, ...]
    figure1_sample_id: int
    figure1_variant: VariantName
    dataset_sample_ids: tuple[int, ...]
    variant_comparison_sample_id: int
    freihand_variant: VariantName
    augmentation_seed: int
    reference_run: str


@dataclass(frozen=True)
class FigureSpec:
    key: str
    label: str
    filename: str
    description: str
    build: Callable[[FigureContext], plt.Figure]


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def _run_label(run_name: str) -> str:
    return RUN_LABELS.get(run_name, run_name)


def _run_colors(run_names: Sequence[str]) -> list[str]:
    return [RUN_COLORS.get(run_name, "#737373") for run_name in run_names]


def _resolve_input_size(config: dict) -> int:
    input_shape = config.get("input_shape")
    if input_shape:
        return int(input_shape[0])
    return DEFAULT_IMAGE_SIZE


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


def discover_run_names() -> list[str]:
    """Return evaluated run names that also have saved config files."""
    run_names: list[str] = []
    for evaluation_path in sorted(ARTIFACTS_DIR.glob("*/evaluation.json")):
        run_name = evaluation_path.parent.name
        if (LOGS_DIR / run_name / "config.json").exists():
            run_names.append(run_name)
    if not run_names:
        raise FileNotFoundError(
            "No evaluated runs found. Expected artifacts/<run>/evaluation.json "
            "and logs/<run>/config.json."
        )
    return run_names


def _summary_by_run(ctx: FigureContext) -> dict[str, dict]:
    return {summary["run_name"]: summary for summary in ctx.summaries}


def _summary_for(ctx: FigureContext, run_name: str) -> dict:
    summaries = _summary_by_run(ctx)
    if run_name in summaries:
        return summaries[run_name]
    return _load_run_summary(run_name)


def _summaries_for(ctx: FigureContext, run_names: Sequence[str]) -> tuple[dict, ...]:
    return tuple(_summary_for(ctx, run_name) for run_name in run_names)


def _model_id(summary: dict) -> str:
    return str(summary["config"].get("model_id", summary["run_name"]))


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


def _load_raw_model(run_name: str) -> keras.Model:
    checkpoint = MODELS_DIR / run_name / "best.keras"
    if not checkpoint.exists():
        raise FileNotFoundError(f"Missing checkpoint: {checkpoint}")
    return keras.models.load_model(str(checkpoint))


def _normalize_keypoints(array: np.ndarray) -> np.ndarray:
    array = np.asarray(array, dtype=np.float32)
    if array.ndim == 2:
        array = array.reshape(array.shape[0], 21, 2)
    return array


def _shape_text(shape) -> str:
    values = [value for value in tuple(shape) if value is not None]
    return " x ".join(str(value) for value in values)


def _apply_report_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": False,
            "axes.axisbelow": True,
            "legend.fontsize": 8,
            "figure.titlesize": 11,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.transparent": False,
        }
    )


def _save_figure(fig: plt.Figure, filename: str, *, output_dir: Path, dpi: int) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / filename
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return path


def _load_sample(sample_id: int, variant: VariantName, *, image_size: int = DEFAULT_IMAGE_SIZE):
    dataset = FreiHand()
    dataset.validate()
    return dataset.sample(
        sample_id,
        variant=variant,
        load_image=True,
        image_size=(image_size, image_size),
        normalize_image=True,
    )


def _format_image_axis(ax, image: np.ndarray, title: str | None = None) -> None:
    ax.imshow(image)
    ax.axis("off")
    if title:
        ax.set_title(title, pad=6)


def _draw_gradient_keypoints(ax, keypoints: np.ndarray) -> None:
    keypoints = np.asarray(keypoints, dtype=np.float32)
    root = keypoints[0]
    distances = np.linalg.norm(keypoints - root, axis=1)
    max_distance = float(distances.max())
    if max_distance <= 0:
        normalized = np.zeros_like(distances)
    else:
        normalized = distances / max_distance

    cmap = plt.get_cmap("turbo")
    colors = [cmap(float(value)) for value in normalized]
    for start, end in HAND_CONNECTIONS:
        ax.plot(
            [keypoints[start, 0], keypoints[end, 0]],
            [keypoints[start, 1], keypoints[end, 1]],
            color=colors[end],
            linewidth=1.6,
            solid_capstyle="round",
        )
    ax.scatter(
        keypoints[:, 0],
        keypoints[:, 1],
        s=19,
        c=colors,
        edgecolors="white",
        linewidths=0.35,
        zorder=3,
    )


def plot_gradient_dataset_sample(
    sample_id: int,
    *,
    variant: VariantName = "gs",
    input_size: int = DEFAULT_IMAGE_SIZE,
) -> plt.Figure:
    sample = _load_sample(sample_id, variant, image_size=input_size)
    fig, ax = plt.subplots(figsize=(4.0, 4.0))
    _format_image_axis(ax, sample.image)
    _draw_gradient_keypoints(ax, sample.keypoints)
    fig.tight_layout(pad=0.05)
    return fig


def plot_dataset_image_row(
    sample_ids: Sequence[int],
    *,
    variant: VariantName = "gs",
    input_size: int = DEFAULT_IMAGE_SIZE,
) -> plt.Figure:
    fig, axes = plt.subplots(1, len(sample_ids), figsize=(2.25 * len(sample_ids), 2.35), squeeze=False)
    for ax, sample_id in zip(axes.flat, sample_ids):
        sample = _load_sample(int(sample_id), variant, image_size=input_size)
        _format_image_axis(ax, sample.image)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.86, bottom=0.02, wspace=0.035)
    return fig


def plot_freihand_variant_comparison(
    sample_id: int,
    *,
    normal_variant: VariantName = "gs",
    freihand_variant: VariantName = "auto",
    input_size: int = DEFAULT_IMAGE_SIZE,
) -> plt.Figure:
    normal = _load_sample(sample_id, normal_variant, image_size=input_size)
    variant = _load_sample(sample_id, freihand_variant, image_size=input_size)
    fig, axes = plt.subplots(1, 2, figsize=(5.2, 2.75), squeeze=False)
    _format_image_axis(axes[0][0], normal.image, "normal")
    _format_image_axis(axes[0][1], variant.image, f"FreiHAND variant: {freihand_variant}")
    fig.subplots_adjust(left=0.02, right=0.98, top=0.85, bottom=0.02, wspace=0.05)
    return fig


def plot_project_augmentation_comparison(
    sample_id: int,
    *,
    variant: VariantName = "gs",
    seed: int = SPLIT_SEED,
    input_size: int = DEFAULT_IMAGE_SIZE,
) -> plt.Figure:
    keras.utils.set_random_seed(seed)
    tf.random.set_seed(seed)
    sample = _load_sample(sample_id, variant, image_size=input_size)
    augmented_image, _ = augment_image_and_keypoints(
        tf.convert_to_tensor(sample.image, dtype=tf.float32),
        tf.convert_to_tensor(sample.keypoints, dtype=tf.float32),
        image_size=input_size,
    )
    fig, axes = plt.subplots(1, 2, figsize=(5.2, 2.75), squeeze=False)
    _format_image_axis(axes[0][0], sample.image, "normal")
    _format_image_axis(axes[0][1], augmented_image.numpy(), "project augmentation")
    fig.subplots_adjust(left=0.02, right=0.98, top=0.85, bottom=0.02, wspace=0.05)
    return fig


def plot_validation_mpke(summaries: Sequence[dict]) -> plt.Figure:
    run_names = [summary["run_name"] for summary in summaries]
    labels = [summary["label"] for summary in summaries]
    values = [float(summary["metrics"]["mpke_px"]) for summary in summaries]
    colors = _run_colors(run_names)

    fig, ax = plt.subplots(figsize=(5.5, 3.4))
    x = np.arange(len(values))
    ax.bar(x, values, color=colors, width=0.58)
    ax.set_xticks(x, labels)
    ax.set_ylabel("MPKE (px)")
    ax.set_title("Validation MPKE (lower is better)")
    ax.set_ylim(0, max(values) * 1.22)
    ax.grid(axis="y", alpha=0.22)
    for index, value in enumerate(values):
        ax.text(index, value + max(values) * 0.025, f"{value:.2f}", ha="center", va="bottom")
    fig.tight_layout()
    return fig


def plot_mpke_distribution(summaries: Sequence[dict]) -> plt.Figure:
    labels = [summary["label"] for summary in summaries]
    x = np.arange(len(summaries), dtype=np.float32)
    width = 0.22
    offsets = (np.arange(len(SUMMARY_METRICS)) - (len(SUMMARY_METRICS) - 1) / 2) * width
    colors = ("#4c78a8", "#72b7b2", "#e45756")

    fig, ax = plt.subplots(figsize=(6.6, 3.9))
    max_value = 0.0
    for offset, (metric_key, metric_label), color in zip(offsets, SUMMARY_METRICS, colors):
        values = [float(summary["metrics"][metric_key]) for summary in summaries]
        max_value = max(max_value, max(values))
        bars = ax.bar(x + offset, values, width=width, label=metric_label, color=color)
        ax.bar_label(bars, labels=[f"{value:.1f}" for value in values], padding=2, fontsize=7.5)

    ax.set_ylabel("Per-sample MPKE (px)")
    ax.set_title("Validation error distribution")
    ax.set_xticks(x, labels)
    ax.set_ylim(0, max_value * 1.28)
    ax.grid(axis="y", alpha=0.22)
    ax.legend(ncols=len(SUMMARY_METRICS), frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.14))
    fig.tight_layout()
    return fig


def plot_training_curves(run_names: Sequence[str]) -> plt.Figure:
    fig, axes = plt.subplots(len(run_names), 2, figsize=(8.0, 2.35 * len(run_names)), squeeze=False)
    for row, run_name in enumerate(run_names):
        history = _load_history(run_name)
        label = _run_label(run_name)
        for col, metric in enumerate(("loss", "mae")):
            ax = axes[row][col]
            train = history.get(metric, [])
            val = history.get(f"val_{metric}", [])
            if train:
                ax.plot(range(1, len(train) + 1), train, label="train", color="#3b6ea8", linewidth=1.7)
            if val:
                ax.plot(range(1, len(val) + 1), val, label="validation", color="#c84d4d", linewidth=1.7)
            ax.set_title(f"{label}: {metric}")
            ax.set_xlabel("epoch")
            ax.set_ylabel(metric)
            ax.grid(axis="y", alpha=0.2)
            ax.legend(frameon=False, loc="best")
    fig.tight_layout()
    return fig


def _architecture_steps(summary: dict, model: keras.Model) -> list[tuple[str, str, str]]:
    model_id = _model_id(summary)
    if model_id == "baseline-model-1":
        return [
            ("Input", f"{_shape_text(model.input_shape[1:])} RGB image", "#eef2f5"),
            ("Conv block 1", "Conv2D 32, BatchNorm, ReLU\nMaxPool", "#dbe9f4"),
            ("Conv block 2", "Conv2D 64, BatchNorm, ReLU\nMaxPool", "#dceee6"),
            ("Conv block 3", "Conv2D 128, BatchNorm, ReLU\nMaxPool", "#f5e5c8"),
            ("Feature pooling", "GlobalAveragePooling2D", "#eadff0"),
            ("Dense head", "Dense 256, Dropout 0.3\nDense 42 linear", "#f3d4d1"),
            ("Output", "21 x (x, y) coordinates", "#eeeeee"),
        ]
    if model_id == "improved-model-1":
        return [
            ("Input", f"{_shape_text(model.input_shape[1:])} RGB image", "#eef2f5"),
            ("Stride-2 stem", "Conv2D 32, BatchNorm, ReLU", "#dbe9f4"),
            ("Residual stage 1", "1 block, 32 filters", "#dceee6"),
            ("Residual stage 2", "2 blocks, 64 filters", "#f5e5c8"),
            ("Residual stage 3", "2 blocks, 128 filters", "#eadff0"),
            ("Heatmap head", "Conv2D 64, BatchNorm, ReLU\n1 x 1 Conv2D 21", "#f3d4d1"),
            ("Output", f"{_shape_text(model.output_shape[1:])} heatmaps", "#eeeeee"),
        ]
    return [
        ("Input", _shape_text(model.input_shape[1:]), "#eef2f5"),
        ("Saved Keras model", f"{len(model.layers)} layers", "#dbe9f4"),
        ("Output", _shape_text(model.output_shape[1:]), "#eeeeee"),
    ]


def plot_model_architecture(summary: dict) -> plt.Figure:
    model = _load_raw_model(summary["run_name"])
    steps = _architecture_steps(summary, model)
    fig, ax = plt.subplots(figsize=(4.6, 7.25))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, len(steps) + 0.75)
    ax.axis("off")
    ax.set_title(f"{summary['label']}\n{model.count_params():,} parameters", pad=8)

    top = len(steps) - 0.05
    for index, (title, detail, color) in enumerate(steps):
        y = top - index
        box = FancyBboxPatch(
            (0.09, y - 0.34),
            0.82,
            0.62,
            boxstyle="round,pad=0.02,rounding_size=0.025",
            linewidth=0.8,
            edgecolor="#6b6b6b",
            facecolor=color,
        )
        ax.add_patch(box)
        ax.text(0.5, y + 0.06, title, ha="center", va="center", fontsize=9, fontweight="bold")
        ax.text(0.5, y - 0.16, detail, ha="center", va="center", fontsize=7.8)
        if index < len(steps) - 1:
            ax.add_patch(
                FancyArrowPatch(
                    (0.5, y - 0.37),
                    (0.5, y - 0.66),
                    arrowstyle="-|>",
                    mutation_scale=9,
                    linewidth=0.8,
                    color="#6b6b6b",
                )
            )

    if _model_id(summary) == "improved-model-1":
        ax.text(
            0.5,
            0.05,
            "Evaluation decodes heatmaps to 21 image-space keypoints.",
            ha="center",
            va="bottom",
            fontsize=7.4,
            color="#555555",
        )
    fig.tight_layout()
    return fig


def _parameter_groups(summary: dict, model: keras.Model) -> list[tuple[str, int]]:
    def layer_params(*prefixes: str) -> int:
        return int(sum(layer.count_params() for layer in model.layers if layer.name.startswith(prefixes)))

    model_id = _model_id(summary)
    if model_id == "baseline-model-1":
        return [
            ("conv block 1", layer_params("conv1", "bn1")),
            ("conv block 2", layer_params("conv2", "bn2")),
            ("conv block 3", layer_params("conv3", "bn3")),
            ("dense head", layer_params("dense_regression", "keypoint_coordinates")),
        ]
    if model_id == "improved-model-1":
        return [
            ("stem", layer_params("stem")),
            ("stage 1", layer_params("stage1")),
            ("stage 2", layer_params("stage2")),
            ("stage 3", layer_params("stage3")),
            ("heatmap head", layer_params("head", "heatmaps")),
        ]

    return [
        (layer.name, int(layer.count_params()))
        for layer in model.layers
        if layer.count_params() > 0
    ]


def _verified_parameter_groups(summary: dict, model: keras.Model) -> list[tuple[str, int]]:
    groups = [(name, count) for name, count in _parameter_groups(summary, model) if count > 0]
    grouped_total = sum(count for _, count in groups)
    model_total = int(model.count_params())
    if grouped_total != model_total:
        raise ValueError(
            f"Parameter groups for {summary['run_name']} sum to {grouped_total:,}, "
            f"but model.count_params() is {model_total:,}."
        )
    return groups


def plot_model_parameter_breakdown(summaries: Sequence[dict]) -> plt.Figure:
    models = {summary["run_name"]: _load_raw_model(summary["run_name"]) for summary in summaries}
    fig, axes = plt.subplots(len(summaries), 1, figsize=(7.6, 2.55 * len(summaries)), squeeze=False)
    colors = ("#3b6ea8", "#72b7b2", "#f0a64d", "#2f8f5b", "#7b5aa6", "#c84d4d")

    for ax, summary in zip(axes.flat, summaries):
        model = models[summary["run_name"]]
        groups = _verified_parameter_groups(summary, model)
        names = [name for name, _ in groups]
        counts = np.asarray([count for _, count in groups], dtype=np.float64)
        y = np.arange(len(groups))

        ax.barh(y, counts, color=[colors[index % len(colors)] for index in range(len(groups))])
        ax.set_yticks(y, names)
        ax.invert_yaxis()
        ax.set_xlabel("parameters")
        ax.set_title(f"{summary['label']}: {model.count_params():,} total parameters")
        ax.grid(axis="x", alpha=0.22)
        ax.grid(axis="y", visible=False)

        max_count = float(counts.max())
        ax.set_xlim(0, max_count * 1.35)
        for index, count in enumerate(counts):
            percent = count / model.count_params() * 100.0
            ax.text(
                count + max_count * 0.025,
                index,
                f"{int(count):,} ({percent:.1f}%)",
                va="center",
                fontsize=8,
            )

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
        ("best", int(representative["best"])),
        ("typical", int(representative["median"])),
        ("worst", int(representative["worst"])),
    ]


def plot_prediction_comparison(summaries: Sequence[dict], *, reference_run: str) -> plt.Figure:
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
    fig, axes = plt.subplots(nrows, ncols, figsize=(2.7 * ncols, 2.45 * nrows), squeeze=False)

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
                gt_color=GT_COLOR,
                pred_color=PRED_COLOR,
                linewidth=1.1,
                marker_size=8,
            )
            if row == 0:
                ax.set_title(f"{difficulty}\nsample {int(sample_id)}", pad=5)
            ax.text(
                0.02,
                0.98,
                f"{summary['label']}\n{errors[col]:.1f} px",
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=7.8,
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.76, "pad": 2.2},
            )
            if row == 0 and col == 0:
                ax.legend(loc="lower right", fontsize=7, framealpha=0.78)

    fig.tight_layout()
    return fig


def generate_figure1(ctx: FigureContext) -> plt.Figure:
    return plot_gradient_dataset_sample(ctx.figure1_sample_id, variant=ctx.figure1_variant)


def generate_figure2(ctx: FigureContext) -> plt.Figure:
    return plot_dataset_image_row(ctx.dataset_sample_ids, variant=ctx.figure1_variant)


def generate_figure3(ctx: FigureContext) -> plt.Figure:
    return plot_freihand_variant_comparison(
        ctx.variant_comparison_sample_id,
        normal_variant="gs",
        freihand_variant=ctx.freihand_variant,
    )


def generate_figure4(ctx: FigureContext) -> plt.Figure:
    return plot_project_augmentation_comparison(
        ctx.variant_comparison_sample_id,
        variant="gs",
        seed=ctx.augmentation_seed,
    )


def generate_figure5(ctx: FigureContext) -> plt.Figure:
    return plot_model_architecture(_summary_for(ctx, "baseline-model-1"))


def generate_figure6(ctx: FigureContext) -> plt.Figure:
    return plot_model_architecture(_summary_for(ctx, "improved-model-1"))


def generate_figure7(ctx: FigureContext) -> plt.Figure:
    return plot_training_curves(ctx.run_names)


def generate_figure8(ctx: FigureContext) -> plt.Figure:
    return plot_validation_mpke(_summaries_for(ctx, ctx.run_names))


def generate_figure9(ctx: FigureContext) -> plt.Figure:
    return plot_mpke_distribution(_summaries_for(ctx, ctx.run_names))


def generate_figure10(ctx: FigureContext) -> plt.Figure:
    return plot_model_parameter_breakdown(_summaries_for(ctx, ARCHITECTURE_RUNS))


def generate_figure11(ctx: FigureContext) -> plt.Figure:
    return plot_prediction_comparison(
        _summaries_for(ctx, QUALITATIVE_RUNS),
        reference_run=ctx.reference_run,
    )


FIGURES: tuple[FigureSpec, ...] = (
    FigureSpec(
        key="1",
        label="Figure 1",
        filename="figure1.png",
        description="FreiHAND sample with ground-truth 2D keypoints and skeleton links colored by distance from the wrist.",
        build=generate_figure1,
    ),
    FigureSpec(
        key="2",
        label="Figure 2",
        filename="figure2.png",
        description="Four FreiHAND training images shown without overlays to illustrate dataset appearance and pose diversity.",
        build=generate_figure2,
    ),
    FigureSpec(
        key="3",
        label="Figure 3",
        filename="figure3.png",
        description="A normal FreiHAND image compared with a FreiHAND-provided processed variant of the same sample.",
        build=generate_figure3,
    ),
    FigureSpec(
        key="4",
        label="Figure 4",
        filename="figure4.png",
        description="A normal FreiHAND image compared with the project's deterministic online augmentation of the same sample.",
        build=generate_figure4,
    ),
    FigureSpec(
        key="5",
        label="Figure 5",
        filename="figure5.png",
        description="Standalone architecture diagram for the baseline coordinate-regression CNN.",
        build=generate_figure5,
    ),
    FigureSpec(
        key="6",
        label="Figure 6",
        filename="figure6.png",
        description="Standalone architecture diagram for the improved residual heatmap CNN.",
        build=generate_figure6,
    ),
    FigureSpec(
        key="7",
        label="Figure 7",
        filename="figure7.png",
        description="Training and validation curves for the baseline, improved, and webcam runs.",
        build=generate_figure7,
    ),
    FigureSpec(
        key="8",
        label="Figure 8",
        filename="figure8.png",
        description="Validation MPKE comparison for the baseline, improved, and webcam runs.",
        build=generate_figure8,
    ),
    FigureSpec(
        key="9",
        label="Figure 9",
        filename="figure9.png",
        description="Median, p90, and p95 per-sample MPKE for the evaluated runs.",
        build=generate_figure9,
    ),
    FigureSpec(
        key="10",
        label="Figure 10",
        filename="figure10.png",
        description="Verified parameter distribution for the two distinct model architectures.",
        build=generate_figure10,
    ),
    FigureSpec(
        key="11",
        label="Figure 11",
        filename="figure11.png",
        description="Qualitative prediction overlays comparing baseline and improved model outputs against ground truth.",
        build=generate_figure11,
    ),
)


def _figure_by_key() -> dict[str, FigureSpec]:
    return {figure.key: figure for figure in FIGURES}


def _normalize_figure_key(value: str) -> str:
    return value.lower().removeprefix("figure").replace(".", "_")


def _parse_figure_selection(values: Sequence[str]) -> tuple[FigureSpec, ...]:
    if not values or values == ["all"]:
        return FIGURES

    by_key = _figure_by_key()
    selected: list[FigureSpec] = []
    for value in values:
        key = _normalize_figure_key(value)
        if key not in by_key:
            valid = ", ".join(figure.key.replace("_", ".") for figure in FIGURES)
            raise argparse.ArgumentTypeError(f"unknown figure {value}; valid figures: {valid}")
        selected.append(by_key[key])
    return tuple(selected)


def _validate_variant(value: str) -> str:
    if value not in VARIANTS:
        raise argparse.ArgumentTypeError(f"invalid variant {value!r}; valid variants: {', '.join(VARIANTS)}")
    return value


def _build_context(args: argparse.Namespace) -> FigureContext:
    run_names = tuple(args.runs or DEFAULT_RUNS)
    summaries = tuple(_load_run_summary(run_name) for run_name in run_names)
    return FigureContext(
        run_names=run_names,
        summaries=summaries,
        figure1_sample_id=args.figure1_sample_id,
        figure1_variant=args.figure1_variant,
        dataset_sample_ids=tuple(args.dataset_sample_ids),
        variant_comparison_sample_id=args.variant_comparison_sample_id,
        freihand_variant=args.freihand_variant,
        augmentation_seed=args.augmentation_seed,
        reference_run=args.reference_run,
    )


def _caption_markdown(figures: Sequence[FigureSpec], output_dir: Path) -> str:
    lines = [
        "# Report figure captions",
        "",
        "Generated by `python -m src.evaluation.report_figures`.",
        "",
    ]
    for figure in figures:
        figure_path = output_dir / figure.filename
        try:
            display_path = figure_path.relative_to(PROJECT_ROOT)
        except ValueError:
            display_path = figure_path
        lines.extend(
            [
                f"## {figure.label}",
                "",
                f"File: `{display_path}`",
                "",
                f"{figure.label}. {figure.description}",
                "",
            ]
        )
    return "\n".join(lines).rstrip() + "\n"


def _write_caption_file(figures: Sequence[FigureSpec], output_dir: Path, caption_output: Path) -> None:
    if not caption_output.is_absolute():
        caption_output = PROJECT_ROOT / caption_output
    caption_output.parent.mkdir(parents=True, exist_ok=True)
    caption_output.write_text(_caption_markdown(figures, output_dir))


def _clean_report_outputs(output_dir: Path) -> None:
    if not output_dir.exists():
        return
    for path in output_dir.glob("figure*.png"):
        path.unlink()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate numbered PNG report figures from saved hand pose runs.",
    )
    parser.add_argument(
        "runs",
        nargs="*",
        help="Run names to include in comparison figures. Defaults to the three evaluated report runs.",
    )
    parser.add_argument(
        "--figures",
        nargs="+",
        help="Figure numbers to generate, such as 1 2 8 11. Defaults to all figures.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_REPORT_FIGURES_DIR,
        help="Directory where numbered PNG figures are written.",
    )
    parser.add_argument(
        "--caption-output",
        type=Path,
        default=DEFAULT_CAPTION_OUTPUT,
        help="Markdown file that stores the figure descriptions.",
    )
    parser.add_argument(
        "--reference-run",
        default="improved-model-1",
        help="Run whose representative validation samples define the qualitative overlay grid.",
    )
    parser.add_argument(
        "--figure1-sample-id",
        type=int,
        default=0,
        help="FreiHAND sample ID used for figure1.",
    )
    parser.add_argument(
        "--figure1-variant",
        type=_validate_variant,
        default="gs",
        help="FreiHAND variant used for figure1 and figure2.",
    )
    parser.add_argument(
        "--dataset-sample-ids",
        type=int,
        nargs="+",
        default=list(DEFAULT_DATASET_SAMPLE_IDS),
        help="FreiHAND sample IDs shown in figure2.",
    )
    parser.add_argument(
        "--variant-comparison-sample-id",
        type=int,
        default=0,
        help="FreiHAND sample ID used for figure3 and figure4.",
    )
    parser.add_argument(
        "--freihand-variant",
        type=_validate_variant,
        default="auto",
        help="FreiHAND-provided variant used in figure3.",
    )
    parser.add_argument(
        "--augmentation-seed",
        type=int,
        default=SPLIT_SEED,
        help="Seed used for deterministic project augmentation in figure4.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="PNG output resolution.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="Print configured figure filenames and descriptions without generating files.",
    )
    parser.add_argument(
        "--no-clean",
        action="store_true",
        help="Do not delete existing figure*.png files before generating the full figure set.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        selected = _parse_figure_selection(args.figures or [])
    except argparse.ArgumentTypeError as exc:
        raise SystemExit(str(exc)) from exc

    if args.list:
        for figure in selected:
            print(f"{figure.label}: {figure.filename} - {figure.description}")
        return

    output_dir = args.output_dir
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir

    if not args.no_clean and selected == FIGURES:
        _clean_report_outputs(output_dir)

    _apply_report_style()
    ctx = _build_context(args)
    generated: list[Path] = []
    for figure in selected:
        generated.append(_save_figure(figure.build(ctx), figure.filename, output_dir=output_dir, dpi=args.dpi))

    _write_caption_file(FIGURES, output_dir, args.caption_output)

    for path in generated:
        print(path.relative_to(PROJECT_ROOT))


if __name__ == "__main__":
    main()
