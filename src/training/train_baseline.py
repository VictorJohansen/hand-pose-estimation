"""Baseline CNN training entry point.

Run from the project root:
    python -m src.training.train_baseline --model-id baseline-model-1
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import keras
import tensorflow as tf

from src.data.freihand import FreiHand, SPLIT_SEED, SPLIT_VALIDATION_FRACTION
from src.models.baseline_cnn import (
    BASELINE_MODEL_1,
    BASELINE_MODEL_2,
    BASELINE_MODEL_IDS,
    DEFAULT_INPUT_SHAPE,
    build_baseline_cnn,
)
from src.training.data_options import add_variant_args, variant_names


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = PROJECT_ROOT / "models"
LOGS_DIR = PROJECT_ROOT / "logs"
DEFAULT_LEARNING_RATES = {
    BASELINE_MODEL_1: 1e-3,
    BASELINE_MODEL_2: 5e-4,
}
DEFAULT_EARLY_STOPPING_PATIENCE = {
    BASELINE_MODEL_1: None,
    BASELINE_MODEL_2: 3,
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the baseline CNN keypoint regressor on FreiHAND.",
    )
    parser.add_argument("--model-id", choices=BASELINE_MODEL_IDS, default=BASELINE_MODEL_1)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--validation-fraction", type=float, default=SPLIT_VALIDATION_FRACTION)
    parser.add_argument("--seed", type=int, default=SPLIT_SEED)
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=None,
        help=(
            "Override early stopping patience. By default baseline-model-2 "
            "uses patience 3 and baseline-model-1 does not use early stopping."
        ),
    )
    parser.add_argument(
        "--limit-train",
        type=int,
        default=None,
        help="Optional cap on training samples (for smoke tests).",
    )
    parser.add_argument(
        "--limit-val",
        type=int,
        default=None,
        help="Optional cap on validation samples.",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Override the default timestamp-based run name.",
    )
    add_variant_args(parser)
    return parser.parse_args(argv)


def configure_logging() -> None:
    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
        level=logging.INFO,
    )


def make_run_name(model_id: str, override: str | None) -> str:
    if override:
        return override
    return model_id


def resolve_learning_rate(args: argparse.Namespace) -> float:
    if args.learning_rate is not None:
        return args.learning_rate
    return DEFAULT_LEARNING_RATES[args.model_id]


def resolve_early_stopping_patience(args: argparse.Namespace) -> int | None:
    if args.early_stopping_patience is not None:
        return args.early_stopping_patience
    return DEFAULT_EARLY_STOPPING_PATIENCE[args.model_id]


def build_datasets(
    args: argparse.Namespace,
    image_size: tuple[int, int],
) -> tuple[tf.data.Dataset, tf.data.Dataset, int, int]:
    dataset = FreiHand()
    dataset.validate()

    train_idx, val_idx = dataset.train_validation_split(
        validation_fraction=args.validation_fraction,
        seed=args.seed,
    )
    if args.limit_train is not None:
        train_idx = train_idx[: args.limit_train]
    if args.limit_val is not None:
        val_idx = val_idx[: args.limit_val]

    train_ds = dataset.tf_dataset(
        indices=train_idx,
        variants=args.train_variants,
        batch_size=args.batch_size,
        image_size=image_size,
        shuffle=True,
        seed=args.seed,
        flatten_keypoints=True,
    )
    val_ds = dataset.tf_dataset(
        indices=val_idx,
        variants=args.val_variants,
        batch_size=args.batch_size,
        image_size=image_size,
        flatten_keypoints=True,
    )
    n_train = int(len(train_idx) * len(args.train_variants))
    n_val = int(len(val_idx) * len(args.val_variants))
    return train_ds, val_ds, n_train, n_val


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    configure_logging()

    keras.utils.set_random_seed(args.seed)

    run_name = make_run_name(args.model_id, args.run_name)
    learning_rate = resolve_learning_rate(args)
    early_stopping_patience = resolve_early_stopping_patience(args)
    model_dir = MODELS_DIR / run_name
    log_dir = LOGS_DIR / run_name
    model_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    image_size = DEFAULT_INPUT_SHAPE[:2]

    logging.info("Run name: %s", run_name)
    logging.info("Train variants: %s", ", ".join(args.train_variants))
    logging.info("Validation variants: %s", ", ".join(args.val_variants))
    logging.info("Building datasets...")
    train_ds, val_ds, n_train, n_val = build_datasets(args, image_size)
    logging.info("Train samples: %d  Val samples: %d", n_train, n_val)

    logging.info("Building %s...", args.model_id)
    model = build_baseline_cnn(input_shape=DEFAULT_INPUT_SHAPE, model_id=args.model_id)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="mse",
        metrics=["mae"],
    )

    checkpoint_path = model_dir / "best.keras"
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            filepath=str(checkpoint_path),
            monitor="val_loss",
            save_best_only=True,
            verbose=1,
        ),
    ]
    if early_stopping_patience is not None:
        callbacks.append(
            keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=early_stopping_patience,
                restore_best_weights=True,
            ),
        )

    config = {
        "run_name": run_name,
        "model_id": args.model_id,
        "model": model.name,
        "representation": "coordinate",
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": learning_rate,
        "early_stopping_patience": early_stopping_patience,
        "validation_fraction": args.validation_fraction,
        "seed": args.seed,
        "n_train": n_train,
        "n_val": n_val,
        "n_train_base_samples": n_train // len(args.train_variants),
        "n_val_base_samples": n_val // len(args.val_variants),
        "train_variants": variant_names(args.train_variants),
        "val_variants": variant_names(args.val_variants),
        "input_shape": list(DEFAULT_INPUT_SHAPE),
        "loss": "mse",
        "metrics": ["mae"],
        "optimizer": "adam",
    }
    (log_dir / "config.json").write_text(json.dumps(config, indent=2))

    logging.info("Training for %d epochs...", args.epochs)
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        callbacks=callbacks,
    )

    history_path = log_dir / "history.json"
    history_path.write_text(json.dumps(history.history, indent=2))

    logging.info("Saved history to %s", history_path)
    logging.info("Best checkpoint: %s", checkpoint_path)


if __name__ == "__main__":
    main()
