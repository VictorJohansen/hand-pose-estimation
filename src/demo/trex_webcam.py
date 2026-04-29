"""Webcam-controlled T-Rex runner demo.

The demo uses a trained hand keypoint model to detect an open hand gesture.
Opening the hand triggers a jump in a small OpenCV runner game. It supports
both coordinate-regression models and heatmap models from this project.

Run from the project root:
    python -m src.demo.trex_webcam --model models/improved-model-1/best.keras

While the final model is still training, run without ``--model``. The script
will use the newest known checkpoint it can find, or fall back to keyboard
control if none exists.
"""

from __future__ import annotations

import argparse
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import cv2
import keras
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CHECKPOINTS = (
    PROJECT_ROOT / "models" / "improved-model-2" / "best.keras",
    PROJECT_ROOT / "models" / "improved-model-1" / "best.keras",
    PROJECT_ROOT / "models" / "baseline-model-1" / "best.keras",
    PROJECT_ROOT / "models" / "baseline-model-2" / "best.keras",
    PROJECT_ROOT / "models" / "improved-aug-smoke" / "best.keras",
)

FINGER_TIPS = (4, 8, 12, 16, 20)
PALM_POINTS = (5, 9, 13, 17)
HAND_CONNECTIONS = (
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Play a small T-Rex runner with webcam hand-pose control.",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=None,
        help="Path to a Keras checkpoint. Auto-detected if omitted.",
    )
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--threshold", type=float, default=1.55)
    parser.add_argument("--camera-width", type=int, default=640)
    parser.add_argument("--camera-height", type=int, default=480)
    parser.add_argument(
        "--no-mirror",
        action="store_true",
        help="Do not mirror the webcam image.",
    )
    parser.add_argument(
        "--keyboard-only",
        action="store_true",
        help="Skip model loading and use Space to jump.",
    )
    parser.add_argument(
        "--strict-model-path",
        action="store_true",
        help="Fail if --model is missing instead of falling back.",
    )
    parser.add_argument(
        "--headless-check",
        action="store_true",
        help="Load the model and run a tiny non-GUI sanity check.",
    )
    return parser.parse_args()


def normalize_checkpoint_path(checkpoint: Path) -> Path:
    if checkpoint.is_absolute():
        return checkpoint
    return PROJECT_ROOT / checkpoint


def available_checkpoints() -> list[Path]:
    return [checkpoint for checkpoint in DEFAULT_CHECKPOINTS if checkpoint.exists()]


def format_available_checkpoints() -> str:
    checkpoints = available_checkpoints()
    if not checkpoints:
        return "none"
    return "\n".join(f"  - {checkpoint_label(path)}" for path in checkpoints)


def find_checkpoint(
    explicit_path: Path | None,
    *,
    strict_model_path: bool = False,
) -> Path | None:
    if explicit_path is not None:
        checkpoint = normalize_checkpoint_path(explicit_path)
        if checkpoint.exists() or strict_model_path:
            return checkpoint
        print(f"Requested checkpoint does not exist: {checkpoint_label(checkpoint)}")
        print("Falling back to the best available checkpoint.")
        print("Available checkpoints:")
        print(format_available_checkpoints())

    checkpoints = available_checkpoints()
    if checkpoints:
        return checkpoints[0]
    return None


def checkpoint_label(checkpoint: Path) -> str:
    try:
        return str(checkpoint.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(checkpoint)


def _is_heatmap_output(output_shape) -> bool:
    return len(output_shape) == 4


class HandPosePredictor:
    """Thin runtime wrapper around project keypoint models."""

    def __init__(self, checkpoint: Path) -> None:
        if not checkpoint.exists():
            raise FileNotFoundError(checkpoint)

        self.checkpoint = checkpoint
        model = keras.models.load_model(str(checkpoint))
        self.raw_output_shape = model.output_shape
        self.input_size = int(model.input_shape[1] or 224)

        if _is_heatmap_output(model.output_shape):
            from src.models.heatmaps import wrap_with_keypoint_decoder

            model = wrap_with_keypoint_decoder(model, input_size=self.input_size)

        self.model = model

    def predict_keypoints(self, frame_bgr: np.ndarray) -> np.ndarray:
        resized = cv2.resize(
            frame_bgr,
            (self.input_size, self.input_size),
            interpolation=cv2.INTER_LINEAR,
        )
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        prediction = self.model.predict_on_batch(rgb[None, ...])
        keypoints = np.asarray(prediction, dtype=np.float32)[0]
        if keypoints.ndim == 1:
            keypoints = keypoints.reshape(21, 2)
        return keypoints


def hand_open_score(keypoints: np.ndarray) -> float:
    """Return a scale-normalized hand openness score from 21 keypoints.

    The score combines finger extension from the wrist and spread between
    adjacent fingertips. It is deliberately simple so it is explainable in the
    semester report and easy to recalibrate during a demo.
    """
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


def scale_keypoints_to_frame(
    keypoints: np.ndarray,
    *,
    source_size: int,
    frame_width: int,
    frame_height: int,
) -> np.ndarray:
    scaled = np.asarray(keypoints, dtype=np.float32).copy()
    scaled[:, 0] *= frame_width / source_size
    scaled[:, 1] *= frame_height / source_size
    return scaled


def draw_keypoints(frame: np.ndarray, keypoints: np.ndarray) -> None:
    keypoints = np.asarray(keypoints, dtype=np.float32).reshape(21, 2)
    for start, end in HAND_CONNECTIONS:
        p1 = tuple(np.round(keypoints[start]).astype(int))
        p2 = tuple(np.round(keypoints[end]).astype(int))
        cv2.line(frame, p1, p2, (0, 255, 110), 2, cv2.LINE_AA)
    for point in keypoints:
        center = tuple(np.round(point).astype(int))
        cv2.circle(frame, center, 4, (40, 40, 255), -1, cv2.LINE_AA)


@dataclass
class Obstacle:
    x: float
    width: int
    height: int


class TrexGame:
    def __init__(self, width: int = 760, height: int = 360) -> None:
        self.width = width
        self.height = height
        self.ground_y = height - 58
        self.dino_x = 76
        self.dino_width = 42
        self.dino_height = 54
        self.reset()

    def reset(self) -> None:
        self.dino_y = float(self.ground_y - self.dino_height)
        self.velocity_y = 0.0
        self.obstacles: list[Obstacle] = []
        self.spawn_timer = 0.0
        self.score = 0.0
        self.speed = 285.0
        self.game_over = False

    @property
    def on_ground(self) -> bool:
        return self.dino_y >= self.ground_y - self.dino_height - 0.1

    def jump(self) -> None:
        if self.game_over:
            self.reset()
            return
        if self.on_ground:
            self.velocity_y = -640.0

    def update(self, dt: float) -> None:
        if self.game_over:
            return

        self.score += dt * 10.0
        self.speed = 285.0 + min(self.score * 1.8, 260.0)

        self.velocity_y += 1750.0 * dt
        self.dino_y += self.velocity_y * dt
        ground_top = self.ground_y - self.dino_height
        if self.dino_y > ground_top:
            self.dino_y = float(ground_top)
            self.velocity_y = 0.0

        self.spawn_timer -= dt
        if self.spawn_timer <= 0.0:
            width = int(np.random.randint(26, 44))
            height = int(np.random.randint(38, 72))
            self.obstacles.append(Obstacle(float(self.width + 20), width, height))
            self.spawn_timer = float(np.random.uniform(0.9, 1.55))

        for obstacle in self.obstacles:
            obstacle.x -= self.speed * dt
        self.obstacles = [
            obstacle for obstacle in self.obstacles if obstacle.x + obstacle.width > 0
        ]

        if self._collides():
            self.game_over = True

    def _collides(self) -> bool:
        dino_left = self.dino_x
        dino_right = self.dino_x + self.dino_width
        dino_top = self.dino_y
        dino_bottom = self.dino_y + self.dino_height

        for obstacle in self.obstacles:
            left = obstacle.x
            right = obstacle.x + obstacle.width
            top = self.ground_y - obstacle.height
            bottom = self.ground_y
            if (
                dino_right > left
                and dino_left < right
                and dino_bottom > top
                and dino_top < bottom
            ):
                return True
        return False

    def render(self, *, gesture_open: bool, score: float, threshold: float) -> np.ndarray:
        canvas = np.full((self.height, self.width, 3), 245, dtype=np.uint8)
        cv2.line(
            canvas,
            (0, self.ground_y),
            (self.width, self.ground_y),
            (85, 85, 85),
            3,
            cv2.LINE_AA,
        )

        dino_top = int(round(self.dino_y))
        dino_bottom = dino_top + self.dino_height
        cv2.rectangle(
            canvas,
            (self.dino_x, dino_top),
            (self.dino_x + self.dino_width, dino_bottom),
            (55, 120, 75),
            -1,
            cv2.LINE_AA,
        )
        cv2.circle(canvas, (self.dino_x + 31, dino_top + 13), 3, (10, 10, 10), -1)
        cv2.rectangle(
            canvas,
            (self.dino_x + 7, dino_bottom - 7),
            (self.dino_x + 17, dino_bottom + 5),
            (55, 120, 75),
            -1,
        )
        cv2.rectangle(
            canvas,
            (self.dino_x + 27, dino_bottom - 7),
            (self.dino_x + 37, dino_bottom + 5),
            (55, 120, 75),
            -1,
        )

        for obstacle in self.obstacles:
            left = int(round(obstacle.x))
            right = left + obstacle.width
            top = self.ground_y - obstacle.height
            cv2.rectangle(
                canvas,
                (left, top),
                (right, self.ground_y),
                (45, 80, 150),
                -1,
                cv2.LINE_AA,
            )

        cv2.putText(
            canvas,
            f"Score {int(self.score):04d}",
            (self.width - 180, 34),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (30, 30, 30),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            f"Hand {'OPEN' if gesture_open else 'closed'}  {score:.2f}/{threshold:.2f}",
            (20, 34),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.67,
            (20, 105, 60) if gesture_open else (80, 80, 80),
            2,
            cv2.LINE_AA,
        )

        if self.game_over:
            cv2.rectangle(canvas, (215, 120), (545, 218), (255, 255, 255), -1)
            cv2.rectangle(canvas, (215, 120), (545, 218), (90, 90, 90), 2)
            cv2.putText(
                canvas,
                "GAME OVER",
                (280, 158),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.05,
                (30, 30, 30),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                canvas,
                "open hand or press R",
                (267, 196),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.62,
                (70, 70, 70),
                1,
                cv2.LINE_AA,
            )

        return canvas


def make_camera_panel(
    frame_bgr: np.ndarray,
    *,
    keypoints: np.ndarray | None,
    input_size: int,
    model_label: str,
    keyboard_only: bool,
) -> np.ndarray:
    panel_width = 390
    panel_height = 360
    panel = np.full((panel_height, panel_width, 3), 32, dtype=np.uint8)

    view = cv2.resize(frame_bgr, (panel_width, 292), interpolation=cv2.INTER_LINEAR)
    if keypoints is not None:
        scaled = scale_keypoints_to_frame(
            keypoints,
            source_size=input_size,
            frame_width=panel_width,
            frame_height=292,
        )
        draw_keypoints(view, scaled)

    panel[:292, :, :] = view
    status = "Keyboard mode" if keyboard_only else "Model mode"
    cv2.putText(
        panel,
        status,
        (14, 318),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.66,
        (235, 235, 235),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        panel,
        model_label[:44],
        (14, 344),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.47,
        (185, 185, 185),
        1,
        cv2.LINE_AA,
    )
    return panel


def render_status_bar(width: int) -> np.ndarray:
    bar = np.full((42, width, 3), 22, dtype=np.uint8)
    cv2.putText(
        bar,
        "Controls: open hand = jump | Space = jump | C = calibrate open hand | [ ] = threshold | R = reset | Q = quit",
        (16, 27),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (230, 230, 230),
        1,
        cv2.LINE_AA,
    )
    return bar


def run_headless_check(predictor: HandPosePredictor | None) -> None:
    game = TrexGame()
    game.update(1.0 / 30.0)
    frame = np.zeros((224, 224, 3), dtype=np.uint8)
    if predictor is not None:
        keypoints = predictor.predict_keypoints(frame)
        score = hand_open_score(keypoints)
        print(
            "headless check ok: "
            f"model={predictor.checkpoint}, "
            f"output_shape={predictor.raw_output_shape}, "
            f"score={score:.3f}"
        )
    else:
        print("headless check ok: no model loaded, keyboard fallback available")


def main() -> None:
    args = parse_args()

    predictor: HandPosePredictor | None = None
    checkpoint = find_checkpoint(
        args.model,
        strict_model_path=args.strict_model_path,
    )
    keyboard_only = args.keyboard_only

    if not keyboard_only and checkpoint is not None:
        if not checkpoint.exists():
            raise SystemExit(
                "Checkpoint does not exist: "
                f"{checkpoint_label(checkpoint)}\n"
                "Available checkpoints:\n"
                f"{format_available_checkpoints()}\n"
                "Remove --strict-model-path to fall back automatically, "
                "or use --keyboard-only to test only the game."
            )
        predictor = HandPosePredictor(checkpoint)
        if "smoke" in checkpoint_label(checkpoint).lower():
            print(
                "Warning: using a smoke-test checkpoint. "
                "This proves the demo wiring, but it is not a trained model."
            )
    elif not keyboard_only:
        keyboard_only = True
        print("No checkpoint found. Starting in keyboard-only mode.")

    if args.headless_check:
        run_headless_check(predictor)
        return

    capture = cv2.VideoCapture(args.camera)
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, args.camera_width)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, args.camera_height)
    if not capture.isOpened():
        raise RuntimeError(f"Could not open webcam index {args.camera}")

    model_label = "no checkpoint"
    input_size = 224
    if predictor is not None:
        model_label = checkpoint_label(predictor.checkpoint)
        input_size = predictor.input_size

    game = TrexGame()
    threshold = float(args.threshold)
    last_time = time.perf_counter()
    was_open = False
    last_jump_time = 0.0
    score = 0.0
    keypoints: np.ndarray | None = None

    window_name = "Hand Pose T-Rex Demo"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if not args.no_mirror:
                frame = cv2.flip(frame, 1)

            now = time.perf_counter()
            dt = min(now - last_time, 0.05)
            last_time = now

            gesture_open = False
            keypoints = None
            if predictor is not None:
                keypoints = predictor.predict_keypoints(frame)
                score = hand_open_score(keypoints)
                gesture_open = score >= threshold
                if gesture_open and not was_open and now - last_jump_time > 0.28:
                    game.jump()
                    last_jump_time = now
                was_open = gesture_open
            else:
                score = 0.0
                was_open = False

            game.update(dt)

            camera_panel = make_camera_panel(
                frame,
                keypoints=keypoints,
                input_size=input_size,
                model_label=model_label,
                keyboard_only=keyboard_only,
            )
            game_panel = game.render(
                gesture_open=gesture_open,
                score=score,
                threshold=threshold,
            )
            body = np.hstack([camera_panel, game_panel])
            display = np.vstack([body, render_status_bar(body.shape[1])])

            cv2.imshow(window_name, display)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key in (ord(" "), ord("w")):
                game.jump()
            elif key == ord("r"):
                game.reset()
            elif key == ord("c") and predictor is not None and math.isfinite(score):
                threshold = max(score * 0.9, 0.1)
                print(f"Calibrated open-hand threshold to {threshold:.3f}")
            elif key == ord("["):
                threshold = max(threshold - 0.05, 0.1)
            elif key == ord("]"):
                threshold += 0.05
    finally:
        capture.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
