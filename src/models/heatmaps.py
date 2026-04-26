"""Heatmap target encoding and decoding for keypoint estimation.

Encoding turns (x, y) keypoints in input-image pixel space into per-keypoint
2D Gaussian heatmaps used as training targets. Decoding inverts that with a
sum-normalized soft-argmax so predictions can be expressed in the same pixel
space as the ground-truth labels and scored with the existing MPKE metric.
"""

from __future__ import annotations

import keras
import tensorflow as tf

DEFAULT_SIGMA = 2.0


def keypoints_to_heatmaps(
    keypoints: tf.Tensor,
    *,
    input_size: int,
    heatmap_size: int,
    sigma: float = DEFAULT_SIGMA,
) -> tf.Tensor:
    """Encode (x, y) keypoints as Gaussian heatmaps.

    keypoints: (B, K, 2) in `input_size` pixel space.
    Returns:    (B, heatmap_size, heatmap_size, K) in [0, 1].
    """
    keypoints = tf.cast(keypoints, tf.float32)
    scale = tf.cast(heatmap_size, tf.float32) / tf.cast(input_size, tf.float32)
    kp = keypoints * scale  # (B, K, 2) in heatmap-pixel space

    side = int(heatmap_size)
    coords = tf.range(side, dtype=tf.float32)
    yy, xx = tf.meshgrid(coords, coords, indexing="ij")  # (H, W)
    xx = xx[None, :, :, None]  # (1, H, W, 1)
    yy = yy[None, :, :, None]

    kp_x = kp[:, None, None, :, 0]  # (B, 1, 1, K)
    kp_y = kp[:, None, None, :, 1]

    sq_dist = (xx - kp_x) ** 2 + (yy - kp_y) ** 2
    return tf.exp(-sq_dist / (2.0 * sigma ** 2))


def heatmaps_to_keypoints(
    heatmaps: tf.Tensor,
    *,
    input_size: int,
) -> tf.Tensor:
    """Decode heatmaps to keypoints with sum-normalized soft-argmax.

    heatmaps: (B, H, W, K), non-negative or near-zero.
    Returns:  (B, K, 2) in `input_size` pixel space.
    """
    heatmaps = tf.cast(heatmaps, tf.float32)
    heatmaps = tf.nn.relu(heatmaps)

    shape = tf.shape(heatmaps)
    batch = shape[0]
    height = shape[1]
    width = shape[2]
    channels = shape[3]

    flat = tf.reshape(heatmaps, [batch, height * width, channels])
    sums = tf.reduce_sum(flat, axis=1, keepdims=True) + 1e-6
    weights = flat / sums  # (B, H*W, K)

    ys = tf.range(height, dtype=tf.float32)
    xs = tf.range(width, dtype=tf.float32)
    yy, xx = tf.meshgrid(ys, xs, indexing="ij")  # (H, W)
    ys_flat = tf.reshape(yy, [height * width, 1])
    xs_flat = tf.reshape(xx, [height * width, 1])

    pred_x = tf.reduce_sum(weights * xs_flat[None, ...], axis=1)  # (B, K)
    pred_y = tf.reduce_sum(weights * ys_flat[None, ...], axis=1)
    keypoints_hm = tf.stack([pred_x, pred_y], axis=-1)  # (B, K, 2)

    scale = tf.cast(input_size, tf.float32) / tf.cast(height, tf.float32)
    return keypoints_hm * scale


@keras.saving.register_keras_serializable(package="hand_pose")
class KeypointDecoder(keras.layers.Layer):
    """Keras layer wrapping `heatmaps_to_keypoints` so a heatmap model can
    be re-exported as a keypoint model and reused with the existing MPKE
    evaluation pipeline."""

    def __init__(self, input_size: int, **kwargs) -> None:
        super().__init__(**kwargs)
        self.input_size = int(input_size)

    def call(self, heatmaps):
        return heatmaps_to_keypoints(heatmaps, input_size=self.input_size)

    def get_config(self):
        config = super().get_config()
        config["input_size"] = self.input_size
        return config


def wrap_with_keypoint_decoder(
    model: keras.Model,
    *,
    input_size: int,
) -> keras.Model:
    """Return a new Keras model that maps images to (B, K, 2) keypoints."""
    keypoints = KeypointDecoder(
        input_size=input_size, name="decoded_keypoints",
    )(model.output)
    return keras.Model(model.input, keypoints, name=f"{model.name}_with_decoder")


__all__ = [
    "DEFAULT_SIGMA",
    "KeypointDecoder",
    "heatmaps_to_keypoints",
    "keypoints_to_heatmaps",
    "wrap_with_keypoint_decoder",
]
