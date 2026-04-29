# Web Deployment

The browser deployment serves the hand-pose model through a FastAPI backend and
uses a browser frontend for webcam access and the T-Rex-style game.

The current desktop demo uses OpenCV to read the local webcam. That does not
work after deployment, because a remote server cannot access a user's laptop
camera. In the web app, the browser reads the webcam, sends compressed frames
to `/api/predict`, and receives keypoints plus the open-hand decision.

## Run Locally

Install the deployment dependencies:

```powershell
venv\Scripts\python.exe -m pip install -r requirements\deploy.txt
```

Run the app:

```powershell
venv\Scripts\python.exe -m uvicorn src.webapp.app:app --host 127.0.0.1 --port 7860
```

Open:

```text
http://127.0.0.1:7860
```

Browser webcam access works on `localhost` during development. For a public
deployment, the app must be served over HTTPS so the browser can grant camera
permission.

The app looks for checkpoints in this order:

1. `MODEL_PATH` environment variable
2. `models/improved-model-2/best.keras`
3. `models/improved-model-1/best.keras`
4. `models/baseline-model-1/best.keras`
5. `models/baseline-model-2/best.keras`
6. `models/improved-aug-smoke/best.keras`

For a specific model:

```powershell
$env:MODEL_PATH="models\improved-model-2\best.keras"
venv\Scripts\python.exe -m uvicorn src.webapp.app:app --host 127.0.0.1 --port 7860
```

## Docker

Build:

```powershell
docker build -t hand-pose-trex .
```

Run:

```powershell
docker run --rm -p 7860:7860 -e MODEL_PATH=models/improved-model-2/best.keras hand-pose-trex
```

Open:

```text
http://127.0.0.1:7860
```

## Hugging Face Spaces

Create a Hugging Face Space using the Docker SDK. The Space repository should
include this project, the `Dockerfile`, and the final Keras checkpoint. If the
checkpoint is too large for normal Git, use Git LFS in the Space repository.

The Space should expose port `7860`, which matches the Dockerfile default.

Set these environment variables if needed:

```text
MODEL_PATH=models/improved-model-2/best.keras
OPEN_HAND_THRESHOLD=1.55
ALLOW_MODEL_FALLBACK=1
```

## Report Wording

Suggested wording:

> The final system was deployed as an interactive web application. The frontend
> uses the browser webcam API to capture live frames and renders a T-Rex-style
> runner game on an HTML canvas. Frames are sent to a FastAPI backend, where
> the trained Keras hand-pose model predicts 21 two-dimensional keypoints. A
> lightweight gesture rule converts the predicted keypoints into an open-hand
> score. When the score exceeds a calibrated threshold, the web game triggers a
> jump. This deployment demonstrates the model as an interactive real-time
> control system rather than only an offline validation model.
