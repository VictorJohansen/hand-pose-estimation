# Web Demo

The browser demo serves the committed hand-pose checkpoint through FastAPI and
uses the browser webcam API for live input. The browser tries MediaPipe hand
detection first, uses its landmarks only to crop the hand, sends the crop to
the project `.keras` model, and maps the returned keypoints back onto the
webcam canvas. If the detector is unavailable or no hand is found, the app
falls back to full-frame inference.

## Run Locally

Create or activate a virtual environment from the repository root, then install
the deployment dependencies:

```bash
python -m pip install -r requirements/deploy.txt
```

Start the service:

```bash
python -m uvicorn src.webapp.app:app --host 127.0.0.1 --port 8080
```

Open:

```text
http://127.0.0.1:8080
```

The MediaPipe crop uses a fixed hand-box scale before resizing the crop for the
model. The `Closed` and `Open` buttons calibrate the gesture threshold from the
current model score.

The default checkpoint is `models/webcam-model/best.keras`. To use a different
checkpoint, set `MODEL_PATH` before starting the service:

```bash
MODEL_PATH=/path/to/model.keras python -m uvicorn src.webapp.app:app --host 127.0.0.1 --port 8080
```

Useful environment variables:

- `MODEL_PATH`: Keras checkpoint path. Defaults to `models/webcam-model/best.keras`.
- `OPEN_HAND_THRESHOLD`: gesture threshold for the open-hand score. Defaults to `1.55`.
- `SERVICE_VERSION`: optional version label returned by `/api/status`.

Smoke-check endpoints:

```bash
curl http://127.0.0.1:8080/healthz
curl http://127.0.0.1:8080/api/status
```

Browser camera access works on `localhost` during development. Public hosting
must use HTTPS for webcam permission.

## Docker

Docker is optional. It packages the FastAPI service and static browser UI into
one image on port `8080`.

```bash
docker build -t hand-pose-estimation:webcam-demo .
docker run --rm -p 8080:8080 hand-pose-estimation:webcam-demo
```
