"""Evaluation metrics for 2D hand keypoint estimation.

Keypoint coordinates are expected as (x, y) pixel positions. Errors are
reported in pixels.

Accepted shapes for keypoint arrays:
    (N, K, 2)   N samples, K keypoints, 2 coords per keypoint
    (N, K*2)    flattened, last dim is (x_0, y_0, x_1, y_1, ...)
"""

from __future__ import annotations

import keras
import numpy as np
import tensorflow as tf


DEFAULT_NUM_KEYPOINTS = 21


def _normalize_keypoints(array: np.ndarray, num_keypoints: int) -> np.ndarray:
    array = np.asarray(array, dtype=np.float32)
    if array.ndim == 2:
        if array.shape[-1] != num_keypoints * 2:
            raise ValueError(
                f"Flattened keypoint array must have last dim {num_keypoints * 2}; "
                f"got {array.shape[-1]}."
            )
        array = array.reshape(array.shape[0], num_keypoints, 2)
    if array.ndim != 3 or array.shape[1] != num_keypoints or array.shape[2] != 2:
        raise ValueError(
            f"Keypoint array must be (N, {num_keypoints}, 2) or "
            f"(N, {num_keypoints * 2}); got shape {array.shape}."
        )
    return array


def per_sample_per_keypoint_error(
    predicted: np.ndarray,
    ground_truth: np.ndarray,
    *,
    num_keypoints: int = DEFAULT_NUM_KEYPOINTS,
) -> np.ndarray:
    """Euclidean pixel distance per (sample, keypoint).

    Returns an array of shape (N, K) in pixels.
    """
    predicted = _normalize_keypoints(predicted, num_keypoints)
    ground_truth = _normalize_keypoints(ground_truth, num_keypoints)
    if predicted.shape != ground_truth.shape:
        raise ValueError(
            f"Shape mismatch: predicted {predicted.shape}, ground_truth {ground_truth.shape}."
        )
    return np.linalg.norm(predicted - ground_truth, axis=-1)


def mean_per_keypoint_error(
    predicted: np.ndarray,
    ground_truth: np.ndarray,
    *,
    num_keypoints: int = DEFAULT_NUM_KEYPOINTS,
) -> float:
    """Mean Euclidean pixel distance over all (sample, keypoint) pairs.

    Single scalar in pixels. Lower is better.
    """
    errors = per_sample_per_keypoint_error(
        predicted, ground_truth, num_keypoints=num_keypoints
    )
    return float(errors.mean())


def per_sample_mpke(
    predicted: np.ndarray,
    ground_truth: np.ndarray,
    *,
    num_keypoints: int = DEFAULT_NUM_KEYPOINTS,
) -> np.ndarray:
    """Per-sample mean Euclidean pixel error.

    Returns an array of shape (N,) where each entry is one sample's mean error
    across its keypoints.
    """
    errors = per_sample_per_keypoint_error(
        predicted, ground_truth, num_keypoints=num_keypoints
    )
    return errors.mean(axis=1)


def mpke_distribution_summary(
    predicted: np.ndarray,
    ground_truth: np.ndarray,
    *,
    num_keypoints: int = DEFAULT_NUM_KEYPOINTS,
    include_representative_indices: bool = False,
) -> dict:
    """Summary statistics of the per-sample MPKE distribution.

    Reports median, p75, p90, p95, and max sample errors in pixels. When
    `include_representative_indices=True`, also returns sample indices for
    the best, median-like, p90-like, and worst samples for use in overlay
    figures.
    """
    sample_errors = per_sample_mpke(
        predicted, ground_truth, num_keypoints=num_keypoints
    )
    summary: dict = {
        "median_sample_mpke_px": float(np.median(sample_errors)),
        "p75_sample_mpke_px": float(np.percentile(sample_errors, 75)),
        "p90_sample_mpke_px": float(np.percentile(sample_errors, 90)),
        "p95_sample_mpke_px": float(np.percentile(sample_errors, 95)),
        "max_sample_mpke_px": float(sample_errors.max()),
    }
    if include_representative_indices:
        order = np.argsort(sample_errors)
        n = len(sample_errors)
        summary["representative_indices"] = {
            "best": int(order[0]),
            "median": int(order[n // 2]),
            "p90": int(order[min(int(n * 0.9), n - 1)]),
            "worst": int(order[-1]),
        }
    return summary


def evaluate_model(
    model: keras.Model,
    dataset: tf.data.Dataset,
    *,
    num_keypoints: int = DEFAULT_NUM_KEYPOINTS,
    include_representative_indices: bool = False,
) -> dict:
    """Run a model over a tf.data dataset and return aggregate metrics.

    The dataset must yield (images, keypoints) batches. Returns the primary
    MPKE plus the per-sample MPKE distribution summary (median, p75, p90, p95,
    max). When `include_representative_indices=True`, also returns sample
    indices for best/median/p90/worst, in the order the dataset yielded them.
    """
    predictions: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    for images, keypoints in dataset:
        predictions.append(np.asarray(model.predict_on_batch(images)))
        targets.append(keypoints.numpy())

    all_predictions = np.concatenate(predictions, axis=0)
    all_targets = np.concatenate(targets, axis=0)

    return {
        "mean_per_keypoint_error_px": mean_per_keypoint_error(
            all_predictions, all_targets, num_keypoints=num_keypoints
        ),
        **mpke_distribution_summary(
            all_predictions,
            all_targets,
            num_keypoints=num_keypoints,
            include_representative_indices=include_representative_indices,
        ),
        "n_samples": int(_normalize_keypoints(all_predictions, num_keypoints).shape[0]),
    }


__all__ = [
    "DEFAULT_NUM_KEYPOINTS",
    "evaluate_model",
    "mean_per_keypoint_error",
    "mpke_distribution_summary",
    "per_sample_mpke",
    "per_sample_per_keypoint_error",
]
