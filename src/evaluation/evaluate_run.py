"""Compute MPKE on the canonical validation split for a saved run.

Persists the result to `artifacts/<run>/evaluation.json` so it can be
referenced by `src.evaluation.comparison` and the report summary without
rerunning evaluation.

Heatmap models are auto-detected by output rank and re-exposed as
keypoint models with `wrap_with_keypoint_decoder` so the existing MPKE
pipeline works unchanged.

Run from the project root:
    python -m src.evaluation.evaluate_run <run-name>
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import keras

from src.data.freihand import FreiHand, SPLIT_SEED, SPLIT_VALIDATION_FRACTION
from src.evaluation.metrics import evaluate_model


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = PROJECT_ROOT / "models"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"


def _is_heatmap_output(output_shape) -> bool:
    return len(output_shape) == 4


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a saved run on the canonical FreiHAND validation split.",
    )
    parser.add_argument("run_name")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--input-size", type=int, default=None)
    parser.add_argument(
        "--limit-val",
        type=int,
        default=None,
        help="Optional cap on validation samples for smoke checks.",
    )
    return parser.parse_args()


def _load_config(run_name: str) -> dict:
    config_path = PROJECT_ROOT / "logs" / run_name / "config.json"
    if not config_path.exists():
        return {}
    return json.loads(config_path.read_text())


def _resolve_input_size(config: dict, override: int | None) -> int:
    if override is not None:
        return override
    input_shape = config.get("input_shape")
    if input_shape:
        return int(input_shape[0])
    return 224


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
        level=logging.INFO,
    )

    config = _load_config(args.run_name)
    input_size = _resolve_input_size(config, args.input_size)

    checkpoint = MODELS_DIR / args.run_name / "best.keras"
    if not checkpoint.exists():
        raise FileNotFoundError(f"No checkpoint at {checkpoint}")

    logging.info("Loading checkpoint %s", checkpoint)
    model = keras.models.load_model(str(checkpoint))
    param_count = int(model.count_params())
    representation = "coordinate"

    if _is_heatmap_output(model.output_shape):
        from src.models.heatmaps import wrap_with_keypoint_decoder
        logging.info(
            "Heatmap output %s detected; wrapping with keypoint decoder.",
            model.output_shape,
        )
        eval_model = wrap_with_keypoint_decoder(model, input_size=input_size)
        representation = "heatmap"
    else:
        eval_model = model

    logging.info("Building canonical validation dataset...")
    dataset = FreiHand()
    dataset.validate()
    _, val_idx = dataset.train_validation_split(
        validation_fraction=SPLIT_VALIDATION_FRACTION,
        seed=SPLIT_SEED,
    )
    if args.limit_val is not None:
        val_idx = val_idx[: args.limit_val]
    val_ds = dataset.tf_dataset(
        indices=val_idx,
        batch_size=args.batch_size,
        image_size=(input_size, input_size),
        flatten_keypoints=True,
    )

    logging.info("Evaluating on %d samples...", int(len(val_idx)))
    metrics = evaluate_model(eval_model, val_ds, include_representative_indices=True)

    payload = {
        "run_name": args.run_name,
        "model_id": config.get("model_id", config.get("model", args.run_name)),
        "representation": config.get("representation", representation),
        "split": {
            "seed": SPLIT_SEED,
            "validation_fraction": SPLIT_VALIDATION_FRACTION,
        },
        "input_size": input_size,
        "param_count": param_count,
        "metrics": metrics,
    }

    out_dir = ARTIFACTS_DIR / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "evaluation.json"
    out_path.write_text(json.dumps(payload, indent=2))
    logging.info("Wrote %s", out_path)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
