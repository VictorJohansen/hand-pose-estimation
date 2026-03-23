from __future__ import annotations

from dataclasses import dataclass, field
from math import ceil
from pathlib import Path
import json
from typing import Iterable, Literal, Sequence

import keras
import numpy as np
import tensorflow as tf
from PIL import Image


VariantName = Literal["gs", "hom", "sample", "auto"]

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ROOT = PROJECT_ROOT / "data" / "FreiHAND_pub_v2"
TRAINING_SAMPLE_COUNT = 32560
IMAGE_SIZE = (224, 224)
VARIANTS: tuple[VariantName, ...] = ("gs", "hom", "sample", "auto")
HAND_CONNECTIONS: tuple[tuple[int, int], ...] = (
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),
    (0, 9),
    (9, 10),
    (10, 11),
    (11, 12),
    (0, 13),
    (13, 14),
    (14, 15),
    (15, 16),
    (0, 17),
    (17, 18),
    (18, 19),
    (19, 20),
)

_PIL_RESAMPLING = getattr(Image, "Resampling", Image)


def _normalize_root(root: str | Path | None) -> Path:
    if root is None:
        return DEFAULT_ROOT

    base = Path(root).expanduser().resolve()
    candidates = (
        base,
        base / "FreiHAND_pub_v2",
        base / "data" / "FreiHAND_pub_v2",
    )
    for candidate in candidates:
        if (candidate / "training" / "rgb").exists() and (candidate / "training_K.json").exists():
            return candidate
    return base


def _normalize_image_size(image_size: int | Sequence[int] | None) -> tuple[int, int]:
    if image_size is None:
        return IMAGE_SIZE
    if isinstance(image_size, int):
        if image_size <= 0:
            raise ValueError("image_size must be positive.")
        return image_size, image_size
    if len(image_size) != 2:
        raise ValueError("image_size must be an int or a (height, width) pair.")
    height, width = int(image_size[0]), int(image_size[1])
    if height <= 0 or width <= 0:
        raise ValueError("image_size values must be positive.")
    return height, width


def _normalize_indices(indices: slice | Iterable[int] | None) -> np.ndarray:
    if indices is None:
        return np.arange(TRAINING_SAMPLE_COUNT, dtype=np.int32)
    if isinstance(indices, slice):
        return np.arange(TRAINING_SAMPLE_COUNT, dtype=np.int32)[indices]

    normalized = np.asarray(list(indices), dtype=np.int32)
    if normalized.ndim != 1:
        raise ValueError("indices must be one-dimensional.")
    if normalized.size == 0:
        return normalized
    if normalized.min() < 0 or normalized.max() >= TRAINING_SAMPLE_COUNT:
        raise IndexError(f"indices must be between 0 and {TRAINING_SAMPLE_COUNT - 1}.")
    return normalized


def _normalize_variants(variants: VariantName | Sequence[VariantName] | str) -> tuple[VariantName, ...]:
    if variants == "all":
        return VARIANTS
    if isinstance(variants, str):
        variants = (variants,)

    normalized = tuple(variants)
    if not normalized:
        raise ValueError("At least one variant must be selected.")

    invalid = [variant for variant in normalized if variant not in VARIANTS]
    if invalid:
        raise ValueError(f"Invalid FreiHAND variants: {invalid}. Valid values: {VARIANTS}.")
    return normalized


def _project_keypoints(xyz: np.ndarray, intrinsics: np.ndarray) -> np.ndarray:
    uvw = np.einsum("nij,nkj->nki", intrinsics, xyz)
    return uvw[..., :2] / uvw[..., 2:3]


@dataclass(slots=True)
class FreiHandSample:
    sample_id: int
    variant: VariantName
    image_id: int
    image_path: Path
    image: np.ndarray | None = None
    keypoints: np.ndarray | None = None


@dataclass(slots=True)
class FreiHand:
    root: str | Path | None = None
    _cache: dict[str, np.ndarray] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        self.root = _normalize_root(self.root)

    @property
    def sample_count(self) -> int:
        return TRAINING_SAMPLE_COUNT

    @property
    def variants(self) -> tuple[VariantName, ...]:
        return VARIANTS

    def validate(self) -> None:
        required_paths = (
            self.root,
            self.root / "training" / "rgb",
            self.root / "training_K.json",
            self.root / "training_xyz.json",
        )
        missing = [str(path) for path in required_paths if not path.exists()]
        if missing:
            raise FileNotFoundError("FreiHAND dataset is missing required paths:\n" + "\n".join(missing))

    def map_image_id(self, sample_id: int, variant: VariantName = "gs") -> int:
        self._validate_sample_id(sample_id)
        variant_name = _normalize_variants(variant)[0]
        return sample_id + TRAINING_SAMPLE_COUNT * VARIANTS.index(variant_name)

    def image_path(self, sample_id: int, variant: VariantName = "gs") -> Path:
        return self.root / "training" / "rgb" / f"{self.map_image_id(sample_id, variant):08d}.jpg"

    def sample(
        self,
        sample_id: int,
        *,
        variant: VariantName = "gs",
        load_image: bool = False,
        image_size: int | Sequence[int] | None = None,
        normalize_image: bool = True,
    ) -> FreiHandSample:
        self._validate_sample_id(sample_id)
        keypoints = self._resize_keypoints(self._uv[sample_id], image_size)
        image_path = self.image_path(sample_id, variant)

        return FreiHandSample(
            sample_id=sample_id,
            variant=variant,
            image_id=self.map_image_id(sample_id, variant),
            image_path=image_path,
            image=self._load_image(image_path, image_size=image_size, normalize=normalize_image) if load_image else None,
            keypoints=keypoints,
        )

    def load_batch(
        self,
        indices: slice | Iterable[int],
        *,
        variants: VariantName | Sequence[VariantName] | str = "gs",
        image_size: int | Sequence[int] | None = None,
        normalize_images: bool = True,
        flatten_keypoints: bool = False,
    ) -> tuple[np.ndarray, np.ndarray]:
        sample_ids = _normalize_indices(indices)
        selected_variants = _normalize_variants(variants)
        if sample_ids.size == 0:
            raise ValueError("No samples selected.")

        image_size = _normalize_image_size(image_size)
        image_paths: list[Path] = []
        repeated_sample_ids: list[int] = []
        for sample_id in sample_ids:
            for variant in selected_variants:
                repeated_sample_ids.append(int(sample_id))
                image_paths.append(self.image_path(int(sample_id), variant))

        images = np.stack(
            [self._load_image(path, image_size=image_size, normalize=normalize_images) for path in image_paths],
            axis=0,
        )
        keypoints = self._resize_keypoints(self._uv[np.asarray(repeated_sample_ids, dtype=np.int32)], image_size)
        if flatten_keypoints:
            keypoints = keypoints.reshape((keypoints.shape[0], -1))
        return images, keypoints.astype(np.float32, copy=False)

    def keras_sequence(
        self,
        *,
        indices: slice | Iterable[int] | None = None,
        variants: VariantName | Sequence[VariantName] | str = "gs",
        batch_size: int = 32,
        shuffle: bool = False,
        seed: int | None = None,
        image_size: int | Sequence[int] | None = None,
        normalize_images: bool = True,
        flatten_keypoints: bool = False,
    ) -> "FreiHandSequence":
        return FreiHandSequence(
            dataset=self,
            indices=indices,
            variants=variants,
            batch_size=batch_size,
            shuffle=shuffle,
            seed=seed,
            image_size=image_size,
            normalize_images=normalize_images,
            flatten_keypoints=flatten_keypoints,
        )

    def tf_dataset(
        self,
        *,
        indices: slice | Iterable[int] | None = None,
        variants: VariantName | Sequence[VariantName] | str = "gs",
        batch_size: int = 32,
        shuffle: bool = False,
        seed: int | None = None,
        image_size: int | Sequence[int] | None = None,
        normalize_images: bool = True,
        flatten_keypoints: bool = False,
        drop_remainder: bool = False,
    ) -> tf.data.Dataset:
        sample_ids = _normalize_indices(indices)
        selected_variants = _normalize_variants(variants)
        if sample_ids.size == 0:
            raise ValueError("No samples selected.")

        image_size = _normalize_image_size(image_size)
        image_paths: list[str] = []
        repeated_sample_ids: list[int] = []
        for sample_id in sample_ids:
            for variant in selected_variants:
                repeated_sample_ids.append(int(sample_id))
                image_paths.append(str(self.image_path(int(sample_id), variant)))

        keypoints = self._resize_keypoints(self._uv[np.asarray(repeated_sample_ids, dtype=np.int32)], image_size)
        if flatten_keypoints:
            keypoints = keypoints.reshape((keypoints.shape[0], -1))

        dataset = tf.data.Dataset.from_tensor_slices((np.asarray(image_paths, dtype=np.str_), keypoints.astype(np.float32)))
        dataset = dataset.map(
            lambda image_path, label: (
                self._decode_image_tf(image_path, image_size, normalize_images),
                label,
            ),
            num_parallel_calls=tf.data.AUTOTUNE,
        )
        if shuffle:
            dataset = dataset.shuffle(buffer_size=len(image_paths), seed=seed, reshuffle_each_iteration=True)
        return dataset.batch(batch_size, drop_remainder=drop_remainder).prefetch(tf.data.AUTOTUNE)

    def train_validation_split(
        self,
        *,
        validation_fraction: float = 0.1,
        shuffle: bool = True,
        seed: int = 42,
    ) -> tuple[np.ndarray, np.ndarray]:
        if not 0.0 < validation_fraction < 1.0:
            raise ValueError("validation_fraction must be between 0 and 1.")

        indices = np.arange(TRAINING_SAMPLE_COUNT, dtype=np.int32)
        if shuffle:
            rng = np.random.default_rng(seed)
            rng.shuffle(indices)

        split_index = int(round((1.0 - validation_fraction) * len(indices)))
        return indices[:split_index], indices[split_index:]

    def plot_sample(
        self,
        sample_id: int,
        *,
        variant: VariantName = "gs",
        image_size: int | Sequence[int] | None = None,
        ax=None,
    ):
        import matplotlib.pyplot as plt

        sample = self.sample(
            sample_id,
            variant=variant,
            load_image=True,
            image_size=image_size,
            normalize_image=True,
        )

        if ax is None:
            _, ax = plt.subplots(figsize=(5, 5))
        ax.imshow(sample.image)
        self._plot_keypoints(ax, sample.keypoints)
        ax.set_title(f"sample={sample_id}, variant={variant}")
        ax.axis("off")
        return ax

    @property
    def _uv(self) -> np.ndarray:
        if "uv" not in self._cache:
            intrinsics = self._load_json_array("training_K.json")
            xyz = self._load_json_array("training_xyz.json")
            self._cache["uv"] = _project_keypoints(xyz, intrinsics).astype(np.float32)
        return self._cache["uv"]

    def _load_json_array(self, filename: str) -> np.ndarray:
        cache_key = filename
        if cache_key in self._cache:
            return self._cache[cache_key]

        path = self.root / filename
        if not path.exists():
            raise FileNotFoundError(path)

        with path.open("r", encoding="utf-8") as file:
            array = np.asarray(json.load(file), dtype=np.float32)

        self._cache[cache_key] = array
        return array

    def _validate_sample_id(self, sample_id: int) -> None:
        if not 0 <= sample_id < TRAINING_SAMPLE_COUNT:
            raise IndexError(f"sample_id must be between 0 and {TRAINING_SAMPLE_COUNT - 1}.")

    def _load_image(
        self,
        image_path: Path,
        *,
        image_size: int | Sequence[int] | None = None,
        normalize: bool = True,
    ) -> np.ndarray:
        target_height, target_width = _normalize_image_size(image_size)
        if not image_path.exists():
            raise FileNotFoundError(image_path)

        with Image.open(image_path) as image:
            image = image.convert("RGB")
            if image.size != (target_width, target_height):
                image = image.resize((target_width, target_height), _PIL_RESAMPLING.BILINEAR)
            array = np.asarray(image, dtype=np.float32)

        if normalize:
            array /= 255.0
        return array

    def _decode_image_tf(
        self,
        image_path: tf.Tensor,
        image_size: tuple[int, int],
        normalize_images: bool,
    ) -> tf.Tensor:
        image = tf.io.read_file(image_path)
        image = tf.image.decode_jpeg(image, channels=3)
        image = tf.cast(image, tf.float32)
        if normalize_images:
            image = image / 255.0
        return tf.image.resize(image, image_size, method="bilinear")

    def _resize_keypoints(self, keypoints: np.ndarray, image_size: int | Sequence[int] | None) -> np.ndarray:
        target_height, target_width = _normalize_image_size(image_size)
        if (target_height, target_width) == IMAGE_SIZE:
            return np.asarray(keypoints, dtype=np.float32).copy()

        scaled = np.asarray(keypoints, dtype=np.float32).copy()
        scaled[..., 0] *= target_width / IMAGE_SIZE[1]
        scaled[..., 1] *= target_height / IMAGE_SIZE[0]
        return scaled

    def _plot_keypoints(self, axis, keypoints: np.ndarray | None) -> None:
        if keypoints is None:
            return
        keypoints = np.asarray(keypoints, dtype=np.float32)
        axis.scatter(keypoints[:, 0], keypoints[:, 1], s=12, c="#00ff66")
        for start, end in HAND_CONNECTIONS:
            axis.plot(
                [keypoints[start, 0], keypoints[end, 0]],
                [keypoints[start, 1], keypoints[end, 1]],
                color="#00ff66",
                linewidth=1.0,
            )


class FreiHandSequence(keras.utils.Sequence):
    def __init__(
        self,
        *,
        dataset: FreiHand,
        indices: slice | Iterable[int] | None = None,
        variants: VariantName | Sequence[VariantName] | str = "gs",
        batch_size: int = 32,
        shuffle: bool = False,
        seed: int | None = None,
        image_size: int | Sequence[int] | None = None,
        normalize_images: bool = True,
        flatten_keypoints: bool = False,
    ) -> None:
        super().__init__()
        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")

        self.dataset = dataset
        self.sample_ids = _normalize_indices(indices)
        self.variants = _normalize_variants(variants)
        if self.sample_ids.size == 0:
            raise ValueError("No samples selected.")

        self.batch_size = batch_size
        self.shuffle = shuffle
        self.normalize_images = normalize_images
        self.flatten_keypoints = flatten_keypoints
        self.image_size = _normalize_image_size(image_size)
        self._order = np.arange(len(self.sample_ids) * len(self.variants), dtype=np.int32)
        self._rng = np.random.default_rng(seed)
        if self.shuffle:
            self.on_epoch_end()

    def __len__(self) -> int:
        return ceil(len(self._order) / self.batch_size)

    def __getitem__(self, index: int) -> tuple[np.ndarray, np.ndarray]:
        start = index * self.batch_size
        stop = min(start + self.batch_size, len(self._order))
        batch_order = self._order[start:stop]

        sample_ids = self.sample_ids[batch_order // len(self.variants)]
        variants = [self.variants[i] for i in batch_order % len(self.variants)]
        image_paths = [self.dataset.image_path(int(sample_id), variant) for sample_id, variant in zip(sample_ids, variants)]

        images = np.stack(
            [self.dataset._load_image(path, image_size=self.image_size, normalize=self.normalize_images) for path in image_paths],
            axis=0,
        )
        keypoints = self.dataset._resize_keypoints(self.dataset._uv[sample_ids], self.image_size)
        if self.flatten_keypoints:
            keypoints = keypoints.reshape((keypoints.shape[0], -1))
        return images, keypoints.astype(np.float32, copy=False)

    def on_epoch_end(self) -> None:
        if self.shuffle:
            self._rng.shuffle(self._order)


__all__ = [
    "FreiHand",
    "FreiHandSample",
    "FreiHandSequence",
    "HAND_CONNECTIONS",
    "IMAGE_SIZE",
    "TRAINING_SAMPLE_COUNT",
    "VARIANTS",
]
