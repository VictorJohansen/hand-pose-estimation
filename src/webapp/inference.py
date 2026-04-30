"""Keras inference helpers for the browser hand-pose demo."""

from __future__ import annotations

import base64
import binascii
import io
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import keras
import numpy as np
from PIL import Image

from src.models.heatmaps import wrap_with_keypoint_decoder


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "webcam-model" / "best.keras"
DEFAULT_THRESHOLD = 1.55
DEFAULT_INPUT_SIZE = 224
FINGER_TIPS = (4, 8, 12, 16, 20)
PALM_POINTS = (5, 9, 13, 17)


@dataclass(frozen=True)
class PredictionResult:
    keypoints: list[list[float]]
    open_score: float
    is_open: bool
    threshold: float
    input_size: int
    model_loaded: bool
    model_path: str
    message: str | None = None


def normalize_model_path(path: str | Path | None) -> Path:
    if path is None or str(path).strip() == "":
        return DEFAULT_MODEL_PATH
    model_path = Path(path)
    if model_path.is_absolute():
        return model_path
    return PROJECT_ROOT / model_path


def path_label(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def output_shape_json(shape: Any) -> Any:
    if isinstance(shape, tuple):
        return [output_shape_json(item) for item in shape]
    if isinstance(shape, list):
        return [output_shape_json(item) for item in shape]
    return shape


def is_heatmap_output(output_shape: Any) -> bool:
    return isinstance(output_shape, tuple) and len(output_shape) == 4


def decode_image_data(image_data: str) -> Image.Image:
    if "," in image_data and image_data.lstrip().startswith("data:"):
        image_data = image_data.split(",", 1)[1]

    try:
        raw = base64.b64decode(image_data, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError("Image payload is not valid base64.") from exc

    try:
        return Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception as exc:
        raise ValueError("Image payload could not be decoded.") from exc


def preprocess_image(image: Image.Image, input_size: int) -> np.ndarray:
    resized = image.resize((input_size, input_size), Image.Resampling.BILINEAR)
    array = np.asarray(resized, dtype=np.float32) / 255.0
    return array[None, ...]


def keypoint_array(prediction: np.ndarray) -> np.ndarray:
    prediction = np.asarray(prediction, dtype=np.float32)
    if prediction.shape == (21, 2):
        return prediction
    if prediction.ndim == 1 and prediction.size == 42:
        return prediction.reshape(21, 2)
    if prediction.ndim == 2 and prediction.shape[-1] == 2:
        return prediction
    raise ValueError(f"Unexpected keypoint output shape: {prediction.shape}")


def hand_open_score(keypoints: np.ndarray) -> float:
    keypoints = np.asarray(keypoints, dtype=np.float32).reshape(21, 2)
    wrist = keypoints[0]
    palm = keypoints[list(PALM_POINTS)]
    tips = keypoints[list(FINGER_TIPS)]

    palm_size = float(np.mean(np.linalg.norm(palm - wrist, axis=1)) + 1e-6)
    extension = float(np.mean(np.linalg.norm(tips - wrist, axis=1)) / palm_size)
    spread = float(
        np.mean(np.linalg.norm(np.diff(tips, axis=0), axis=1)) / palm_size,
    )
    return 0.65 * extension + 0.35 * spread


class HandPoseService:
    """Loads one Keras checkpoint and serves single-frame predictions."""

    def __init__(
        self,
        *,
        model_path: str | Path | None = None,
        threshold: float = DEFAULT_THRESHOLD,
        service_version: str | None = None,
    ) -> None:
        self.threshold = float(threshold)
        self.checkpoint = normalize_model_path(model_path)
        self.service_version = service_version or "local"
        self.model = None
        self.raw_output_shape: Any = None
        self.input_size = DEFAULT_INPUT_SIZE
        self.load_message: str | None = None

        self._load_model()

    @classmethod
    def from_environment(cls) -> "HandPoseService":
        return cls(
            model_path=os.environ.get("MODEL_PATH"),
            threshold=float(os.environ.get("OPEN_HAND_THRESHOLD", DEFAULT_THRESHOLD)),
            service_version=os.environ.get("SERVICE_VERSION"),
        )

    @property
    def model_loaded(self) -> bool:
        return self.model is not None

    @property
    def model_path_label(self) -> str:
        return path_label(self.checkpoint)

    def _load_model(self) -> None:
        if not self.checkpoint.exists():
            self.load_message = f"Checkpoint does not exist: {self.model_path_label}"
            return

        try:
            model = keras.models.load_model(str(self.checkpoint), compile=False)
            self.raw_output_shape = model.output_shape
            self.input_size = int(model.input_shape[1] or DEFAULT_INPUT_SIZE)
            if is_heatmap_output(model.output_shape):
                model = wrap_with_keypoint_decoder(model, input_size=self.input_size)
            self.model = model
            self.load_message = None
        except Exception as exc:
            self.model = None
            self.load_message = f"Checkpoint could not be loaded: {exc}"

    def status(self) -> dict[str, Any]:
        return {
            "serviceVersion": self.service_version,
            "modelLoaded": self.model_loaded,
            "modelPath": self.model_path_label,
            "inputSize": self.input_size,
            "threshold": self.threshold,
            "message": self.load_message,
            "rawOutputShape": output_shape_json(self.raw_output_shape),
        }

    def predict_data_url(
        self,
        image_data: str,
        *,
        threshold: float | None = None,
    ) -> PredictionResult:
        active_threshold = float(self.threshold if threshold is None else threshold)
        if self.model is None:
            return PredictionResult(
                keypoints=[],
                open_score=0.0,
                is_open=False,
                threshold=active_threshold,
                input_size=self.input_size,
                model_loaded=False,
                model_path=self.model_path_label,
                message=self.load_message or "No model loaded.",
            )

        image = decode_image_data(image_data)
        batch = preprocess_image(image, self.input_size)
        prediction = np.asarray(self.model.predict_on_batch(batch), dtype=np.float32)[0]
        keypoints = keypoint_array(prediction)
        score = hand_open_score(keypoints)

        return PredictionResult(
            keypoints=np.round(keypoints, 3).astype(float).tolist(),
            open_score=score,
            is_open=score >= active_threshold,
            threshold=active_threshold,
            input_size=self.input_size,
            model_loaded=True,
            model_path=self.model_path_label,
            message=self.load_message,
        )

