"""Train-time image/keypoint augmentations for hand pose estimation."""

from __future__ import annotations

import math

import tensorflow as tf


DEFAULT_MAX_TRANSLATION_FRACTION = 0.05
DEFAULT_MAX_ROTATION_DEGREES = 15.0
DEFAULT_SCALE_MIN = 0.9
DEFAULT_SCALE_MAX = 1.1
DEFAULT_BRIGHTNESS_DELTA = 0.10
DEFAULT_CONTRAST_MIN = 0.9
DEFAULT_CONTRAST_MAX = 1.1
DEFAULT_SATURATION_MIN = 0.9
DEFAULT_SATURATION_MAX = 1.1


def _translation_matrix(tx: tf.Tensor, ty: tf.Tensor) -> tf.Tensor:
    tx = tf.cast(tx, tf.float32)
    ty = tf.cast(ty, tf.float32)
    one = tf.ones_like(tx)
    zero = tf.zeros_like(tx)
    row1 = tf.stack([one, zero, tx], axis=-1)
    row2 = tf.stack([zero, one, ty], axis=-1)
    row3 = tf.stack([zero, zero, one], axis=-1)
    return tf.stack([row1, row2, row3], axis=-2)


def _scale_matrix(scale: tf.Tensor) -> tf.Tensor:
    scale = tf.cast(scale, tf.float32)
    zero = tf.zeros_like(scale)
    one = tf.ones_like(scale)
    row1 = tf.stack([scale, zero, zero], axis=-1)
    row2 = tf.stack([zero, scale, zero], axis=-1)
    row3 = tf.stack([zero, zero, one], axis=-1)
    return tf.stack([row1, row2, row3], axis=-2)


def _rotation_matrix(angle_radians: tf.Tensor) -> tf.Tensor:
    cosine = tf.cos(angle_radians)
    sine = tf.sin(angle_radians)
    zero = tf.zeros_like(cosine)
    one = tf.ones_like(cosine)
    row1 = tf.stack([cosine, -sine, zero], axis=-1)
    row2 = tf.stack([sine, cosine, zero], axis=-1)
    row3 = tf.stack([zero, zero, one], axis=-1)
    return tf.stack([row1, row2, row3], axis=-2)


def _compose_affine_transform(
    *,
    image_size: int,
    tx: tf.Tensor,
    ty: tf.Tensor,
    angle_radians: tf.Tensor,
    scale: tf.Tensor,
) -> tf.Tensor:
    """Create a forward input-to-output transform around the image center."""
    center = (tf.cast(image_size, tf.float32) - 1.0) / 2.0
    to_center = _translation_matrix(-center, -center)
    from_center = _translation_matrix(center, center)
    transform = (
        _translation_matrix(tx, ty)
        @ from_center
        @ _rotation_matrix(angle_radians)
        @ _scale_matrix(scale)
        @ to_center
    )
    return transform


def _matrix_to_projective_vector(matrix: tf.Tensor) -> tf.Tensor:
    """Convert a 3x3 affine matrix into TensorFlow's 8-value transform form."""
    return tf.stack(
        [
            matrix[..., 0, 0],
            matrix[..., 0, 1],
            matrix[..., 0, 2],
            matrix[..., 1, 0],
            matrix[..., 1, 1],
            matrix[..., 1, 2],
            matrix[..., 2, 0],
            matrix[..., 2, 1],
        ],
        axis=-1,
    )


def _apply_affine_to_keypoints(
    keypoints: tf.Tensor,
    transform: tf.Tensor,
    image_size: int,
) -> tf.Tensor:
    keypoints = tf.cast(keypoints, tf.float32)
    ones = tf.ones(tf.concat([tf.shape(keypoints)[:-1], [1]], axis=0), dtype=tf.float32)
    homogeneous = tf.concat([keypoints, ones], axis=-1)
    transformed = tf.linalg.matmul(homogeneous, transform, transpose_b=True)[..., :2]

    max_coord = tf.cast(image_size - 1, tf.float32)
    x = tf.clip_by_value(transformed[..., 0], 0.0, max_coord)
    y = tf.clip_by_value(transformed[..., 1], 0.0, max_coord)
    return tf.stack([x, y], axis=-1)


def augmentation_config() -> dict[str, float]:
    """Return the default online augmentation parameters for run configs."""
    return {
        "max_translation_fraction": DEFAULT_MAX_TRANSLATION_FRACTION,
        "max_rotation_degrees": DEFAULT_MAX_ROTATION_DEGREES,
        "scale_min": DEFAULT_SCALE_MIN,
        "scale_max": DEFAULT_SCALE_MAX,
        "brightness_delta": DEFAULT_BRIGHTNESS_DELTA,
        "contrast_min": DEFAULT_CONTRAST_MIN,
        "contrast_max": DEFAULT_CONTRAST_MAX,
        "saturation_min": DEFAULT_SATURATION_MIN,
        "saturation_max": DEFAULT_SATURATION_MAX,
    }


def augment_image_and_keypoints(
    image: tf.Tensor,
    keypoints: tf.Tensor,
    *,
    image_size: int,
    max_translation_fraction: float = DEFAULT_MAX_TRANSLATION_FRACTION,
    max_rotation_degrees: float = DEFAULT_MAX_ROTATION_DEGREES,
    scale_min: float = DEFAULT_SCALE_MIN,
    scale_max: float = DEFAULT_SCALE_MAX,
    brightness_delta: float = DEFAULT_BRIGHTNESS_DELTA,
    contrast_min: float = DEFAULT_CONTRAST_MIN,
    contrast_max: float = DEFAULT_CONTRAST_MAX,
    saturation_min: float = DEFAULT_SATURATION_MIN,
    saturation_max: float = DEFAULT_SATURATION_MAX,
) -> tuple[tf.Tensor, tf.Tensor]:
    """Apply conservative train-time augmentation to image/keypoint tensors."""
    image = tf.cast(image, tf.float32)
    keypoints = tf.cast(keypoints, tf.float32)
    is_unbatched = image.shape.rank == 3

    if is_unbatched:
        image = image[None, ...]
        keypoints = keypoints[None, ...]

    batch_size = tf.shape(image)[0]

    max_translation_pixels = tf.cast(image_size, tf.float32) * tf.cast(max_translation_fraction, tf.float32)
    tx = tf.random.uniform([batch_size], -max_translation_pixels, max_translation_pixels)
    ty = tf.random.uniform([batch_size], -max_translation_pixels, max_translation_pixels)
    angle_radians = tf.random.uniform([batch_size], -max_rotation_degrees, max_rotation_degrees)
    angle_radians = angle_radians * (math.pi / 180.0)
    scale = tf.random.uniform([batch_size], scale_min, scale_max)

    forward_transform = _compose_affine_transform(
        image_size=image_size,
        tx=tx,
        ty=ty,
        angle_radians=angle_radians,
        scale=scale,
    )
    inverse_transform = tf.linalg.inv(forward_transform)
    projective_transform = _matrix_to_projective_vector(inverse_transform)

    augmented_image = tf.raw_ops.ImageProjectiveTransformV3(
        images=image,
        transforms=projective_transform,
        output_shape=tf.constant([image_size, image_size], dtype=tf.int32),
        interpolation="BILINEAR",
        fill_value=0.0,
    )
    augmented_keypoints = _apply_affine_to_keypoints(keypoints, forward_transform, image_size)

    augmented_image = tf.image.random_brightness(augmented_image, max_delta=brightness_delta)
    augmented_image = tf.image.random_contrast(augmented_image, lower=contrast_min, upper=contrast_max)
    augmented_image = tf.image.random_saturation(augmented_image, lower=saturation_min, upper=saturation_max)
    augmented_image = tf.clip_by_value(augmented_image, 0.0, 1.0)

    if is_unbatched:
        augmented_image = augmented_image[0]
        augmented_keypoints = augmented_keypoints[0]

    return augmented_image, augmented_keypoints


__all__ = [
    "DEFAULT_BRIGHTNESS_DELTA",
    "DEFAULT_CONTRAST_MAX",
    "DEFAULT_CONTRAST_MIN",
    "DEFAULT_MAX_ROTATION_DEGREES",
    "DEFAULT_MAX_TRANSLATION_FRACTION",
    "DEFAULT_SATURATION_MAX",
    "DEFAULT_SATURATION_MIN",
    "DEFAULT_SCALE_MAX",
    "DEFAULT_SCALE_MIN",
    "augmentation_config",
    "augment_image_and_keypoints",
]
