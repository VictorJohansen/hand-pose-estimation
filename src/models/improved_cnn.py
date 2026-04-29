"""Improved model CNN with residual blocks and a heatmap head."""

from __future__ import annotations

import keras
from keras import layers

DEFAULT_INPUT_SHAPE: tuple[int, int, int] = (224, 224, 3)
DEFAULT_NUM_KEYPOINTS = 21
DEFAULT_HEATMAP_SIZE = 56


def _residual_block(
    x,
    filters: int,
    stride: int = 1,
    name: str = "res",
):
    shortcut = x
    in_channels = x.shape[-1]
    if stride != 1 or in_channels != filters:
        shortcut = layers.Conv2D(
            filters, 1, strides=stride, padding="same", use_bias=False,
            name=f"{name}_proj_conv",
        )(x)
        shortcut = layers.BatchNormalization(name=f"{name}_proj_bn")(shortcut)

    x = layers.Conv2D(
        filters, 3, strides=stride, padding="same", use_bias=False,
        name=f"{name}_conv1",
    )(x)
    x = layers.BatchNormalization(name=f"{name}_bn1")(x)
    x = layers.ReLU(name=f"{name}_relu1")(x)

    x = layers.Conv2D(
        filters, 3, padding="same", use_bias=False,
        name=f"{name}_conv2",
    )(x)
    x = layers.BatchNormalization(name=f"{name}_bn2")(x)

    x = layers.Add(name=f"{name}_add")([x, shortcut])
    x = layers.ReLU(name=f"{name}_relu2")(x)
    return x


def build_improved_cnn(
    input_shape: tuple[int, int, int] = DEFAULT_INPUT_SHAPE,
    num_keypoints: int = DEFAULT_NUM_KEYPOINTS,
    heatmap_size: int = DEFAULT_HEATMAP_SIZE,
    output_activation: str = "sigmoid",
) -> keras.Model:
    inputs = keras.Input(shape=input_shape, name="input_image")

    x = layers.Conv2D(
        32, 3, strides=2, padding="same", use_bias=False, name="stem_conv",
    )(inputs)
    x = layers.BatchNormalization(name="stem_bn")(x)
    x = layers.ReLU(name="stem_relu")(x)

    x = _residual_block(x, 32, stride=1, name="stage1_block1")

    x = _residual_block(x, 64, stride=2, name="stage2_block1")
    x = _residual_block(x, 64, stride=1, name="stage2_block2")

    x = _residual_block(x, 128, stride=1, name="stage3_block1")
    x = _residual_block(x, 128, stride=1, name="stage3_block2")

    x = layers.Conv2D(64, 3, padding="same", use_bias=False, name="head_conv")(x)
    x = layers.BatchNormalization(name="head_bn")(x)
    x = layers.ReLU(name="head_relu")(x)

    outputs = layers.Conv2D(
        num_keypoints, 1, padding="same",
        activation=output_activation,
        name="heatmaps",
    )(x)

    model = keras.Model(inputs, outputs, name="residual_heatmap_cnn")

    expected_hw = (heatmap_size, heatmap_size)
    actual_hw = tuple(model.output_shape[1:3])
    if actual_hw != expected_hw:
        raise ValueError(
            f"Heatmap spatial size {actual_hw} does not match expected {expected_hw}. "
            "Adjust input_shape or strides so the head emits heatmap_size x heatmap_size.",
        )

    return model
