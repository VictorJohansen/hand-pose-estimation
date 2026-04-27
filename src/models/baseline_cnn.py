"""Baseline CNN regressors for 2D hand keypoint estimation."""

from __future__ import annotations

import keras
from keras import layers

DEFAULT_INPUT_SHAPE: tuple[int, int, int] = (224, 224, 3)
DEFAULT_NUM_KEYPOINTS = 21
BASELINE_MODEL_1 = "baseline-model-1"
BASELINE_MODEL_2 = "baseline-model-2"
BASELINE_MODEL_IDS = (BASELINE_MODEL_1, BASELINE_MODEL_2)


def build_baseline_model_1(
    input_shape: tuple[int, int, int] = DEFAULT_INPUT_SHAPE,
    num_keypoints: int = DEFAULT_NUM_KEYPOINTS,
) -> keras.Model:
    """Build the simple coordinate-regression baseline.

    This matches the architecture used for the tracked baseline-model-1
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

    return keras.Model(inputs, outputs, name=BASELINE_MODEL_1)


def build_baseline_model_2(
    input_shape: tuple[int, int, int] = DEFAULT_INPUT_SHAPE,
    num_keypoints: int = DEFAULT_NUM_KEYPOINTS,
) -> keras.Model:
    """Build the regularized coordinate-regression baseline.

    This standardizes Mikal's notebook baseline to the same 224x224 input
    protocol used by the other report models.
    """
    inputs = keras.Input(shape=input_shape, name="input_image")

    x = layers.Conv2D(32, 3, padding="same", activation="relu", name="block1_conv1")(inputs)
    x = layers.BatchNormalization(name="block1_bn")(x)
    x = layers.Conv2D(32, 3, padding="same", activation="relu", name="block1_conv2")(x)
    x = layers.MaxPooling2D(name="block1_pool")(x)

    x = layers.Conv2D(64, 3, padding="same", activation="relu", name="block2_conv1")(x)
    x = layers.BatchNormalization(name="block2_bn")(x)
    x = layers.Conv2D(64, 3, padding="same", activation="relu", name="block2_conv2")(x)
    x = layers.MaxPooling2D(name="block2_pool")(x)

    x = layers.Conv2D(128, 3, padding="same", activation="relu", name="block3_conv")(x)
    x = layers.MaxPooling2D(name="block3_pool")(x)

    x = layers.Flatten(name="flatten_spatial_features")(x)
    x = layers.Dense(128, activation="relu", name="dense_regression")(x)
    x = layers.Dropout(0.5, name="dropout_regularization")(x)

    outputs = layers.Dense(
        num_keypoints * 2,
        activation="linear",
        name="keypoint_coordinates",
    )(x)

    return keras.Model(inputs, outputs, name=BASELINE_MODEL_2)


def build_baseline_cnn(
    input_shape: tuple[int, int, int] = DEFAULT_INPUT_SHAPE,
    num_keypoints: int = DEFAULT_NUM_KEYPOINTS,
    model_id: str = BASELINE_MODEL_1,
) -> keras.Model:
    if model_id == BASELINE_MODEL_1:
        return build_baseline_model_1(input_shape, num_keypoints)
    if model_id == BASELINE_MODEL_2:
        return build_baseline_model_2(input_shape, num_keypoints)
    raise ValueError(f"Unknown baseline model_id '{model_id}'. Expected one of {BASELINE_MODEL_IDS}.")


__all__ = [
    "BASELINE_MODEL_1",
    "BASELINE_MODEL_2",
    "BASELINE_MODEL_IDS",
    "DEFAULT_INPUT_SHAPE",
    "DEFAULT_NUM_KEYPOINTS",
    "build_baseline_cnn",
    "build_baseline_model_1",
    "build_baseline_model_2",
]
