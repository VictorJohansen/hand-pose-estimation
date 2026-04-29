"""Generate numbered PNG figures for the project report.

Run from the project root:
    python -m src.evaluation.report_figure_set
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from src.evaluation import report_figures


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "reports" / "figures"
DEFAULT_CAPTION_OUTPUT = PROJECT_ROOT / "docs" / "report-figure-captions.md"
DEFAULT_RUNS = (
    "baseline-model-1",
    "improved-model-1",
    "improved-model-1-online-augmented",
)


@dataclass(frozen=True)
class FigureContext:
    run_names: tuple[str, ...]
    summaries: tuple[dict, ...]
    dataset_samples: tuple[int, ...]
    reference_run: str


@dataclass(frozen=True)
class FigureSpec:
    number: int
    title: str
    description: str
    build: Callable[[FigureContext], plt.Figure]

    @property
    def filename(self) -> str:
        return f"figure{self.number}.png"

    @property
    def report_label(self) -> str:
        return f"Figure {self.number}"


def generate_figure1(ctx: FigureContext) -> plt.Figure:
    return report_figures.plot_dataset_examples(ctx.dataset_samples)


def generate_figure2(ctx: FigureContext) -> plt.Figure:
    return report_figures.plot_primary_results(ctx.summaries)


def generate_figure3(ctx: FigureContext) -> plt.Figure:
    return report_figures.plot_error_summary(ctx.summaries)


def generate_figure4(ctx: FigureContext) -> plt.Figure:
    return report_figures.plot_training_curves(ctx.run_names)


def generate_figure5(ctx: FigureContext) -> plt.Figure:
    return report_figures.plot_model_architectures(ctx.summaries)


def generate_figure6(ctx: FigureContext) -> plt.Figure:
    return report_figures.plot_model_parameter_breakdown(ctx.summaries)


def generate_figure7(ctx: FigureContext) -> plt.Figure:
    return report_figures.plot_prediction_comparison(
        ctx.summaries,
        reference_run=ctx.reference_run,
    )


FIGURES: tuple[FigureSpec, ...] = (
    FigureSpec(
        number=1,
        title="FreiHAND dataset examples",
        description=(
            "Examples from the FreiHAND training split with projected "
            "ground-truth 2D hand keypoints."
        ),
        build=generate_figure1,
    ),
    FigureSpec(
        number=2,
        title="Primary validation results",
        description=(
            "Validation mean per-keypoint error and model parameter count for "
            "the compared runs."
        ),
        build=generate_figure2,
    ),
    FigureSpec(
        number=3,
        title="Per-sample error distribution",
        description=(
            "Median, p75, p90, and p95 per-sample MPKE on the canonical "
            "validation split."
        ),
        build=generate_figure3,
    ),
    FigureSpec(
        number=4,
        title="Training curves",
        description="Training and validation loss/MAE curves for each run.",
        build=generate_figure4,
    ),
    FigureSpec(
        number=5,
        title="Model architecture overview",
        description=(
            "Compact architecture schematics for the coordinate-regression "
            "baseline and heatmap-based improved model."
        ),
        build=generate_figure5,
    ),
    FigureSpec(
        number=6,
        title="Parameter breakdown",
        description="Trainable parameter distribution by major model block.",
        build=generate_figure6,
    ),
    FigureSpec(
        number=7,
        title="Prediction comparison",
        description=(
            "Representative validation predictions compared with ground-truth "
            "keypoints for the evaluated models."
        ),
        build=generate_figure7,
    ),
)


def _figure_by_number() -> dict[int, FigureSpec]:
    return {figure.number: figure for figure in FIGURES}


def _parse_figure_selection(values: Sequence[str]) -> tuple[FigureSpec, ...]:
    if not values or values == ["all"]:
        return FIGURES

    by_number = _figure_by_number()
    selected: list[FigureSpec] = []
    for value in values:
        normalized = value.lower().removeprefix("figure")
        try:
            number = int(normalized)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(f"invalid figure selection: {value}") from exc
        if number not in by_number:
            valid = ", ".join(str(figure.number) for figure in FIGURES)
            raise argparse.ArgumentTypeError(f"unknown figure {number}; valid figures: {valid}")
        selected.append(by_number[number])
    return tuple(selected)


def _build_context(args: argparse.Namespace) -> FigureContext:
    run_names = tuple(args.runs)
    summaries = tuple(report_figures._load_run_summary(run_name) for run_name in run_names)
    return FigureContext(
        run_names=run_names,
        summaries=summaries,
        dataset_samples=tuple(args.dataset_samples),
        reference_run=args.reference_run,
    )


def _save_png(fig: plt.Figure, path: Path, *, dpi: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def _caption_markdown(figures: Sequence[FigureSpec], output_dir: Path) -> str:
    lines = [
        "# Report figure captions",
        "",
        "Generated by `python -m src.evaluation.report_figure_set`.",
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
                f"## {figure.report_label}",
                "",
                f"File: `{display_path}`",
                "",
                f"{figure.report_label}. {figure.description}",
                "",
            ]
        )
    return "\n".join(lines).rstrip() + "\n"


def _write_caption_file(figures: Sequence[FigureSpec], output_dir: Path, caption_output: Path) -> None:
    if not caption_output.is_absolute():
        caption_output = PROJECT_ROOT / caption_output
    caption_output.parent.mkdir(parents=True, exist_ok=True)
    caption_output.write_text(_caption_markdown(figures, output_dir))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate numbered PNG report figures.",
    )
    parser.add_argument(
        "figures",
        nargs="*",
        help="Figure numbers to generate, such as 1 3 7. Defaults to all figures.",
    )
    parser.add_argument(
        "--runs",
        nargs="+",
        default=list(DEFAULT_RUNS),
        help="Run names to include in model comparison figures.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
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
        default="baseline-model-1",
        help="Run whose representative validation samples define the overlay grid.",
    )
    parser.add_argument(
        "--dataset-samples",
        type=int,
        nargs="+",
        default=list(report_figures.DEFAULT_DATASET_SAMPLES),
        help="FreiHAND training sample IDs to show in figure1.",
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
        help="Print configured figure numbers and descriptions without generating files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        selected = _parse_figure_selection(args.figures)
    except argparse.ArgumentTypeError as exc:
        raise SystemExit(str(exc)) from exc
    if args.list:
        for figure in selected:
            print(f"{figure.report_label}: {figure.filename} - {figure.description}")
        return

    report_figures._apply_report_style()
    output_dir = args.output_dir
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir

    ctx = _build_context(args)
    generated: list[Path] = []
    for figure in selected:
        output_path = output_dir / figure.filename
        _save_png(figure.build(ctx), output_path, dpi=args.dpi)
        generated.append(output_path)

    _write_caption_file(selected, output_dir, args.caption_output)

    for path in generated:
        print(path.relative_to(PROJECT_ROOT))


if __name__ == "__main__":
    main()
