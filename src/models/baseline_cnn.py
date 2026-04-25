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
    inputs = keras.Input(shape=input_shape)

    x = layers.Conv2D(32, 3, padding="same")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D(2)(x)

    x = layers.Conv2D(64, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D(2)(x)

    x = layers.Conv2D(128, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D(2)(x)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_keypoints * 2)(x)

    return keras.Model(inputs, outputs, name="baseline_cnn")
