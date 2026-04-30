"""FastAPI entry point for the browser hand-pose demo."""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from src.webapp.inference import HandPoseService


PROJECT_ROOT = Path(__file__).resolve().parents[2]
STATIC_DIR = PROJECT_ROOT / "src" / "webapp" / "static"


class PredictRequest(BaseModel):
    image: str = Field(..., description="Base64 JPEG crop or data URL.")
    threshold: float | None = Field(None, gt=0.0, le=10.0)


service = HandPoseService.from_environment()

app = FastAPI(
    title="Hand Pose Browser Demo",
    version="0.1.0",
    description="Browser webcam demo backed by the DAT255 hand-pose model.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.get("/healthz")
def healthz() -> dict:
    return {"ok": service.model_loaded, **service.status()}


@app.get("/api/status")
def status() -> dict:
    return service.status()


@app.post("/api/predict")
def predict(request: PredictRequest) -> dict:
    try:
        result = service.predict_data_url(
            request.image,
            threshold=request.threshold,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return {
        "keypoints": result.keypoints,
        "openScore": result.open_score,
        "isOpen": result.is_open,
        "threshold": result.threshold,
        "inputSize": result.input_size,
        "modelLoaded": result.model_loaded,
        "modelPath": result.model_path,
        "message": result.message,
        "status": service.status(),
    }


app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")
