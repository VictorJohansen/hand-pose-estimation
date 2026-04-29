"""Compare two or more runs and emit a markdown comparison table.

Reads `artifacts/<run>/evaluation.json` for each run and uses
`logs/<run>/config.json` when it is available. The resulting table is intended
to be pasted into the report with minimal editing.

Run from the project root:
    python -m src.evaluation.comparison baseline-model improved-model
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
LOGS_DIR = PROJECT_ROOT / "logs"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"

MODEL_DISPLAY_NAMES = {
    "baseline-model": "baseline model",
    "improved-model": "improved model",
    "webcam-model": "webcam model",
}

DEFAULT_COLUMNS: tuple[tuple[str, str], ...] = (
    ("display_name", "Model"),
    ("representation", "Repr."),
    ("param_count", "Params"),
    ("mpke_px", "MPKE (px)"),
    ("median_sample_mpke_px", "Median (px)"),
    ("p90_sample_mpke_px", "p90 (px)"),
    ("p95_sample_mpke_px", "p95 (px)"),
    ("max_sample_mpke_px", "Max (px)"),
)


def _infer_representation(model_id: str) -> str:
    if model_id in {"improved-model", "webcam-model"}:
        return "heatmap"
    if model_id == "baseline-model":
        return "coordinate"
    return ""


def load_run_summary(run_name: str) -> dict:
    config_path = LOGS_DIR / run_name / "config.json"
    eval_path = ARTIFACTS_DIR / run_name / "evaluation.json"
    if not eval_path.exists():
        raise FileNotFoundError(
            f"Missing evaluation for run '{run_name}'. "
            f"Run `python -m src.evaluation.evaluate_run {run_name}` first."
        )
    config = json.loads(config_path.read_text()) if config_path.exists() else {}
    evaluation = json.loads(eval_path.read_text())
    display_name = MODEL_DISPLAY_NAMES.get(run_name, run_name)
    model_id = config.get("model_id", evaluation.get("model_id", run_name))
    return {
        "run_name": run_name,
        "display_name": display_name,
        "model_id": model_id,
        "model": config.get("model", "baseline_cnn"),
        "representation": config.get(
            "representation",
            evaluation.get("representation", _infer_representation(model_id)),
        ),
        "epochs": config.get("epochs"),
        "batch_size": config.get("batch_size"),
        "learning_rate": config.get("learning_rate"),
        "param_count": evaluation.get("param_count"),
        "metrics": evaluation.get("metrics", {}),
    }


def _format_value(value) -> str:
    if value is None:
        return "—"
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, float):
        return f"{value:.2f}"
    if isinstance(value, int) and abs(value) >= 1000:
        return f"{value:,}"
    return str(value)


def format_comparison_markdown(
    summaries: list[dict],
    columns: tuple[tuple[str, str], ...] = DEFAULT_COLUMNS,
) -> str:
    headers = [label for _, label in columns]
    rendered_rows: list[list[str]] = []
    for summary in summaries:
        row: list[str] = []
        for key, _ in columns:
            value = summary.get(key)
            if value is None:
                value = summary.get("metrics", {}).get(key)
            row.append(_format_value(value))
        rendered_rows.append(row)

    widths = [
        max(len(headers[i]), *(len(r[i]) for r in rendered_rows))
        for i in range(len(headers))
    ]

    def render(row: list[str]) -> str:
        return "| " + " | ".join(c.ljust(widths[i]) for i, c in enumerate(row)) + " |"

    lines = [
        render(headers),
        "| " + " | ".join("-" * w for w in widths) + " |",
    ]
    for row in rendered_rows:
        lines.append(render(row))
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare two or more runs as a markdown table.",
    )
    parser.add_argument("runs", nargs="+", help="Run names to include in the comparison.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summaries = [load_run_summary(name) for name in args.runs]
    print(format_comparison_markdown(summaries))


if __name__ == "__main__":
    main()


__all__ = [
    "DEFAULT_COLUMNS",
    "MODEL_DISPLAY_NAMES",
    "format_comparison_markdown",
    "load_run_summary",
]
