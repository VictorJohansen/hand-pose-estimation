"""Keras inference helpers for the browser-based hand pose app."""

from __future__ import annotations

import base64
import io
import os
from dataclasses import dataclass
from pathlib import Path

import keras
import numpy as np
from PIL import Image

from src.models.heatmaps import wrap_with_keypoint_decoder


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_CANDIDATES = (
    PROJECT_ROOT / "models" / "improved-model-2" / "best.keras",
    PROJECT_ROOT / "models" / "improved-model-1" / "best.keras",
    PROJECT_ROOT / "models" / "baseline-model-1" / "best.keras",
    PROJECT_ROOT / "models" / "baseline-model-2" / "best.keras",
    PROJECT_ROOT / "models" / "improved-aug-smoke" / "best.keras",
)

FINGER_TIPS = (4, 8, 12, 16, 20)
PALM_POINTS = (5, 9, 13, 17)
DEFAULT_THRESHOLD = 1.55


@dataclass(frozen=True)
class PredictionResult:
    keypoints: list[list[float]]
    open_score: float
    is_open: bool
    threshold: float
    input_size: int
    model_loaded: bool
    model_path: str | None
    message: str | None = None


def normalize_model_path(path: str | Path | None) -> Path | None:
    if path is None or str(path).strip() == "":
        return None
    normalized = Path(path)
    if normalized.is_absolute():
        return normalized
    return PROJECT_ROOT / normalized


def path_label(path: Path | None) -> str | None:
    if path is None:
        return None
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def first_available_model(preferred: Path | None) -> Path | None:
    candidates: list[Path] = []
    if preferred is not None:
        candidates.append(preferred)
    candidates.extend(DEFAULT_MODEL_CANDIDATES)

    seen: set[Path] = set()
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        if candidate.exists():
            return candidate
    return None


def is_heatmap_output(output_shape) -> bool:
    return len(output_shape) == 4


def decode_image_data(image_data: str) -> Image.Image:
    if "," in image_data and image_data.lstrip().startswith("data:"):
        image_data = image_data.split(",", 1)[1]
    try:
        raw = base64.b64decode(image_data, validate=True)
    except Exception as exc:
        raise ValueError("Image payload is not valid base64.") from exc

    try:
        return Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception as exc:
        raise ValueError("Image payload could not be decoded.") from exc


def preprocess_image(image: Image.Image, input_size: int) -> np.ndarray:
    resized = image.resize((input_size, input_size), Image.Resampling.BILINEAR)
    array = np.asarray(resized, dtype=np.float32) / 255.0
    return array[None, ...]


def hand_open_score(keypoints: np.ndarray) -> float:
    keypoints = np.asarray(keypoints, dtype=np.float32).reshape(21, 2)
    wrist = keypoints[0]
    palm = keypoints[list(PALM_POINTS)]
    tips = keypoints[list(FINGER_TIPS)]

    palm_size = float(np.mean(np.linalg.norm(palm - wrist, axis=1)) + 1e-6)
    extension = float(np.mean(np.linalg.norm(tips - wrist, axis=1)) / palm_size)
    spread = float(
        np.mean(np.linalg.norm(np.diff(tips, axis=0), axis=1)) / palm_size
    )
    return 0.65 * extension + 0.35 * spread


class HandPoseService:
    """Loads a Keras hand pose checkpoint and serves single-frame predictions."""

    def __init__(
        self,
        *,
        model_path: str | Path | None = None,
        threshold: float = DEFAULT_THRESHOLD,
        allow_fallback: bool = True,
    ) -> None:
        preferred = normalize_model_path(model_path)
        checkpoint = first_available_model(preferred) if allow_fallback else preferred

        self.threshold = float(threshold)
        self.checkpoint = checkpoint
        self.model = None
        self.raw_output_shape = None
        self.input_size = 224
        self.load_message: str | None = None

        if checkpoint is None:
            self.load_message = "No Keras checkpoint found."
            return
        if not checkpoint.exists():
            self.load_message = f"Configured checkpoint does not exist: {path_label(checkpoint)}"
            return

        model = keras.models.load_model(str(checkpoint), compile=False)
        self.raw_output_shape = model.output_shape
        self.input_size = int(model.input_shape[1] or 224)
        if is_heatmap_output(model.output_shape):
            model = wrap_with_keypoint_decoder(model, input_size=self.input_size)
        self.model = model

        if "smoke" in str(checkpoint).lower():
            self.load_message = (
                "Smoke checkpoint loaded. This proves deployment wiring, "
                "but it is not a final trained model."
            )

    @classmethod
    def from_environment(cls) -> "HandPoseService":
        threshold = float(os.environ.get("OPEN_HAND_THRESHOLD", DEFAULT_THRESHOLD))
        fallback = os.environ.get("ALLOW_MODEL_FALLBACK", "1").lower()
        allow_fallback = fallback not in {"0", "false", "no"}
        return cls(
            model_path=os.environ.get("MODEL_PATH"),
            threshold=threshold,
            allow_fallback=allow_fallback,
        )

    @property
    def model_loaded(self) -> bool:
        return self.model is not None

    @property
    def model_path_label(self) -> str | None:
        return path_label(self.checkpoint)

    def status(self) -> dict:
        return {
            "modelLoaded": self.model_loaded,
            "modelPath": self.model_path_label,
            "inputSize": self.input_size,
            "threshold": self.threshold,
            "message": self.load_message,
            "rawOutputShape": self.raw_output_shape,
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
        if prediction.ndim == 1:
            prediction = prediction.reshape(21, 2)

        score = hand_open_score(prediction)
        return PredictionResult(
            keypoints=np.round(prediction, 3).astype(float).tolist(),
            open_score=score,
            is_open=score >= active_threshold,
            threshold=active_threshold,
            input_size=self.input_size,
            model_loaded=True,
            model_path=self.model_path_label,
            message=self.load_message,
        )

