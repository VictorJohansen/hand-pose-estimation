"""Baseline model CNN regressors for 2D hand keypoint estimation."""

from __future__ import annotations

import keras
from keras import layers

DEFAULT_INPUT_SHAPE: tuple[int, int, int] = (224, 224, 3)
DEFAULT_NUM_KEYPOINTS = 21
BASELINE_MODEL = "baseline-model"
BASELINE_MODEL_IDS = (BASELINE_MODEL,)


def build_baseline_model(
    input_shape: tuple[int, int, int] = DEFAULT_INPUT_SHAPE,
    num_keypoints: int = DEFAULT_NUM_KEYPOINTS,
) -> keras.Model:
    """Build the simple coordinate-regression baseline model.

    This matches the architecture used for the tracked baseline-model
    evaluation artifact.
    """
    inputs = keras.Input(shape=input_shape, name="input_image")

    x = layers.Conv2D(32, 3, padding="same", name="conv1")(inputs)
    x = layers.BatchNormalization(name="bn1")(x)
    x = layers.ReLU(name="relu1")(x)
    x = layers.MaxPooling2D(2, name="pool1")(x)

    x = layers.Conv2D(64, 3, padding="same", name="conv2")(x)
    x = layers.BatchNormalization(name="bn2")(x)
    x = layers.ReLU(name="relu2")(x)
    x = layers.MaxPooling2D(2, name="pool2")(x)

    x = layers.Conv2D(128, 3, padding="same", name="conv3")(x)
    x = layers.BatchNormalization(name="bn3")(x)
    x = layers.ReLU(name="relu3")(x)
    x = layers.MaxPooling2D(2, name="pool3")(x)

    x = layers.GlobalAveragePooling2D(name="global_average_pool")(x)

    x = layers.Dense(256, activation="relu", name="dense_regression")(x)
    x = layers.Dropout(0.3, name="dropout_regularization")(x)

    outputs = layers.Dense(
        num_keypoints * 2,
        activation="linear",
        name="keypoint_coordinates",
    )(x)

    return keras.Model(inputs, outputs, name=BASELINE_MODEL)


def build_baseline_cnn(
    input_shape: tuple[int, int, int] = DEFAULT_INPUT_SHAPE,
    num_keypoints: int = DEFAULT_NUM_KEYPOINTS,
    model_id: str = BASELINE_MODEL,
) -> keras.Model:
    if model_id == BASELINE_MODEL:
        return build_baseline_model(input_shape, num_keypoints)
    raise ValueError(f"Unknown baseline model_id '{model_id}'. Expected one of {BASELINE_MODEL_IDS}.")


__all__ = [
    "BASELINE_MODEL",
    "BASELINE_MODEL_IDS",
    "DEFAULT_INPUT_SHAPE",
    "DEFAULT_NUM_KEYPOINTS",
    "build_baseline_cnn",
    "build_baseline_model",
]
