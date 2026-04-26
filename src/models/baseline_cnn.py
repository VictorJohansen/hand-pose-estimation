"""Baseline CNN regressor for 2D hand keypoint estimation."""

from __future__ import annotations

import keras
from keras import layers

DEFAULT_INPUT_SHAPE: tuple[int, int, int] = (224, 224, 3)
DEFAULT_NUM_KEYPOINTS = 21


def build_baseline_cnn(
    input_shape: tuple[int, int, int] = DEFAULT_INPUT_SHAPE,
    num_keypoints: int = DEFAULT_NUM_KEYPOINTS,
) -> keras.Model:
    inputs = keras.Input(shape=input_shape, name="input_image")

    x = layers.Conv2D(32, 3, padding="same", activation="relu", name="conv1")(inputs)
    x = layers.MaxPooling2D(2, name="pool1")(x)

    x = layers.Conv2D(64, 3, padding="same", activation="relu", name="conv2")(x)
    x = layers.MaxPooling2D(2, name="pool2")(x)

    x = layers.Conv2D(128, 3, padding="same", activation="relu", name="conv3")(x)
    x = layers.MaxPooling2D(2, name="pool3")(x)

    x = layers.Conv2D(128, 3, padding="same", activation="relu", name="conv4")(x)
    x = layers.MaxPooling2D(2, name="pool4")(x)

    x = layers.Flatten(name="flatten_spatial_features")(x)

    x = layers.Dense(256, activation="relu", name="dense_regression")(x)
    x = layers.Dropout(0.3, name="dropout_regularization")(x)

    outputs = layers.Dense(
        num_keypoints * 2,
        activation="linear",
        name="keypoint_coordinates",
    )(x)

    return keras.Model(inputs, outputs, name="baseline_cnn")
