const video = document.getElementById("video");
const cameraCanvas = document.getElementById("cameraCanvas");
const cameraCtx = cameraCanvas.getContext("2d");
const gameCanvas = document.getElementById("gameCanvas");
const gameCtx = gameCanvas.getContext("2d");

const cameraButton = document.getElementById("cameraButton");
const calibrateClosedButton = document.getElementById("calibrateClosedButton");
const calibrateOpenButton = document.getElementById("calibrateOpenButton");
const resetButton = document.getElementById("resetButton");
const modelStatus = document.getElementById("modelStatus");
const detectorStatus = document.getElementById("detectorStatus");
const cropMode = document.getElementById("cropMode");
const gestureState = document.getElementById("gestureState");
const gameState = document.getElementById("gameState");
const thresholdSlider = document.getElementById("thresholdSlider");
const thresholdValue = document.getElementById("thresholdValue");
const scoreValue = document.getElementById("scoreValue");
const openScoreValue = document.getElementById("openScoreValue");

const MEDIAPIPE_VERSION = "0.10.35";
const MEDIAPIPE_PACKAGE =
  `https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@${MEDIAPIPE_VERSION}`;
const MEDIAPIPE_WASM = `${MEDIAPIPE_PACKAGE}/wasm`;
const HAND_MODEL_URL =
  "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task";

const HAND_CONNECTIONS = [
  [0, 1], [1, 2], [2, 3], [3, 4],
  [0, 5], [5, 6], [6, 7], [7, 8],
  [0, 9], [9, 10], [10, 11], [11, 12],
  [0, 13], [13, 14], [14, 15], [15, 16],
  [0, 17], [17, 18], [18, 19], [19, 20],
];

const captureCanvas = document.createElement("canvas");
const captureCtx = captureCanvas.getContext("2d");
const CROP_SCALE = 1.8;
const CROP_PADDING = 56;
const DISPLAY_KEYPOINT_SCALE = 1.25;

const detector = {
  handLandmarker: null,
  loading: false,
  ready: false,
  failed: false,
};

let cameraReady = false;
let latestPrediction = null;
let latestCanvasKeypoints = [];
let latestCrop = null;
let latestScore = 0;
let threshold = Number(thresholdSlider.value);
let modelInputSize = 224;
let calibratedClosedScore = null;
let calibratedOpenScore = null;
let previousOpenState = null;
let predictionBusy = false;
let lastPredictionAt = 0;
let lastFrameAt = performance.now();

const game = {
  groundY: 302,
  runnerX: 76,
  runnerY: 248,
  runnerWidth: 42,
  runnerHeight: 54,
  velocityY: 0,
  speed: 285,
  score: 0,
  spawnTimer: 0,
  obstacles: [],
  gameOver: false,
};

function resetGame() {
  game.runnerY = game.groundY - game.runnerHeight;
  game.velocityY = 0;
  game.speed = 285;
  game.score = 0;
  game.spawnTimer = 0;
  game.obstacles = [];
  game.gameOver = false;
}

function jump() {
  if (game.gameOver) {
    resetGame();
    return;
  }
  if (game.runnerY >= game.groundY - game.runnerHeight - 0.1) {
    game.velocityY = -640;
  }
}

function updateGame(dt) {
  if (game.gameOver) {
    return;
  }

  game.score += dt * 10;
  game.speed = 285 + Math.min(game.score * 1.8, 260);

  game.velocityY += 1750 * dt;
  game.runnerY += game.velocityY * dt;
  const groundTop = game.groundY - game.runnerHeight;
  if (game.runnerY > groundTop) {
    game.runnerY = groundTop;
    game.velocityY = 0;
  }

  game.spawnTimer -= dt;
  if (game.spawnTimer <= 0) {
    game.obstacles.push({
      x: gameCanvas.width + 24,
      width: 24 + Math.floor(Math.random() * 18),
      height: 36 + Math.floor(Math.random() * 36),
    });
    game.spawnTimer = 0.9 + Math.random() * 0.65;
  }

  for (const obstacle of game.obstacles) {
    obstacle.x -= game.speed * dt;
  }
  game.obstacles = game.obstacles.filter(
    (obstacle) => obstacle.x + obstacle.width > 0,
  );

  if (collides()) {
    game.gameOver = true;
  }
}

function collides() {
  const runnerLeft = game.runnerX;
  const runnerRight = game.runnerX + game.runnerWidth;
  const runnerTop = game.runnerY;
  const runnerBottom = game.runnerY + game.runnerHeight;

  return game.obstacles.some((obstacle) => {
    const left = obstacle.x;
    const right = obstacle.x + obstacle.width;
    const top = game.groundY - obstacle.height;
    const bottom = game.groundY;
    return (
      runnerRight > left &&
      runnerLeft < right &&
      runnerBottom > top &&
      runnerTop < bottom
    );
  });
}

function drawGame() {
  gameCtx.clearRect(0, 0, gameCanvas.width, gameCanvas.height);
  gameCtx.fillStyle = "#f8faf7";
  gameCtx.fillRect(0, 0, gameCanvas.width, gameCanvas.height);

  gameCtx.strokeStyle = "#5b625e";
  gameCtx.lineWidth = 3;
  gameCtx.beginPath();
  gameCtx.moveTo(0, game.groundY);
  gameCtx.lineTo(gameCanvas.width, game.groundY);
  gameCtx.stroke();

  gameCtx.fillStyle = "#2e8754";
  gameCtx.fillRect(game.runnerX, game.runnerY, game.runnerWidth, game.runnerHeight);
  gameCtx.fillStyle = "#14201a";
  gameCtx.beginPath();
  gameCtx.arc(game.runnerX + 31, game.runnerY + 13, 3, 0, Math.PI * 2);
  gameCtx.fill();
  gameCtx.fillStyle = "#2e8754";
  gameCtx.fillRect(game.runnerX + 7, game.runnerY + game.runnerHeight - 7, 10, 12);
  gameCtx.fillRect(game.runnerX + 27, game.runnerY + game.runnerHeight - 7, 10, 12);

  gameCtx.fillStyle = "#2f6fa3";
  for (const obstacle of game.obstacles) {
    gameCtx.fillRect(
      obstacle.x,
      game.groundY - obstacle.height,
      obstacle.width,
      obstacle.height,
    );
  }

  if (game.gameOver) {
    gameCtx.fillStyle = "rgba(255, 255, 255, 0.92)";
    gameCtx.fillRect(226, 122, 308, 94);
    gameCtx.strokeStyle = "#5b625e";
    gameCtx.strokeRect(226, 122, 308, 94);
    gameCtx.fillStyle = "#1e2522";
    gameCtx.font = "30px Arial";
    gameCtx.fillText("GAME OVER", 284, 161);
    gameCtx.font = "18px Arial";
    gameCtx.fillText("reset or open hand", 302, 196);
  }
}

function drawCamera() {
  cameraCtx.fillStyle = "#101614";
  cameraCtx.fillRect(0, 0, cameraCanvas.width, cameraCanvas.height);

  if (cameraReady) {
    cameraCtx.save();
    cameraCtx.translate(cameraCanvas.width, 0);
    cameraCtx.scale(-1, 1);
    cameraCtx.drawImage(video, 0, 0, cameraCanvas.width, cameraCanvas.height);
    cameraCtx.restore();
  }

  if (latestCrop) {
    drawCropBox(latestCrop);
  }
  if (latestCanvasKeypoints.length) {
    drawKeypoints(latestCanvasKeypoints);
  }
}

function drawCropBox(crop) {
  if (crop.mode !== "detector") {
    return;
  }
  const x = cameraCanvas.width -
    ((crop.x + crop.width) / crop.sourceWidth) * cameraCanvas.width;
  const y = (crop.y / crop.sourceHeight) * cameraCanvas.height;
  const width = (crop.width / crop.sourceWidth) * cameraCanvas.width;
  const height = (crop.height / crop.sourceHeight) * cameraCanvas.height;

  cameraCtx.strokeStyle = "rgba(255, 198, 83, 0.95)";
  cameraCtx.lineWidth = 2;
  cameraCtx.strokeRect(x, y, width, height);
}

function drawKeypoints(points) {
  cameraCtx.lineWidth = 2;
  cameraCtx.strokeStyle = "#17d677";
  for (const [start, end] of HAND_CONNECTIONS) {
    if (!points[start] || !points[end]) {
      continue;
    }
    cameraCtx.beginPath();
    cameraCtx.moveTo(points[start].x, points[start].y);
    cameraCtx.lineTo(points[end].x, points[end].y);
    cameraCtx.stroke();
  }

  cameraCtx.fillStyle = "#ef476f";
  for (const point of points) {
    cameraCtx.beginPath();
    cameraCtx.arc(point.x, point.y, 4, 0, Math.PI * 2);
    cameraCtx.fill();
  }
}

function updateReadouts() {
  scoreValue.textContent = String(Math.floor(game.score)).padStart(4, "0");
  openScoreValue.textContent = latestScore.toFixed(2);
  thresholdValue.textContent = threshold.toFixed(2);
  gameState.textContent = game.gameOver ? "over" : "running";

  const isOpen = Boolean(latestPrediction?.isOpen);
  gestureState.textContent = isOpen ? "open" : "closed";
  gestureState.classList.toggle("open", isOpen);
}

function updateDetectorStatus() {
  if (detector.ready) {
    detectorStatus.textContent = "ready";
  } else if (detector.failed) {
    detectorStatus.textContent = "fallback";
  } else {
    detectorStatus.textContent = "loading";
  }
}

async function refreshStatus() {
  const response = await fetch("/api/status");
  const status = await response.json();
  const modelName = status.modelPath || "no checkpoint";
  const note = status.message ? ` | ${status.message}` : "";
  modelStatus.textContent =
    `${status.modelLoaded ? "Model loaded" : "Model missing"}: ${modelName}${note}`;
  modelInputSize = Number(status.inputSize || modelInputSize);
  threshold = Number(status.threshold || threshold);
  thresholdSlider.value = threshold.toFixed(2);
  updateReadouts();
}

async function createHandLandmarker(HandLandmarker, vision, delegate) {
  return HandLandmarker.createFromOptions(vision, {
    baseOptions: {
      modelAssetPath: HAND_MODEL_URL,
      delegate,
    },
    runningMode: "VIDEO",
    numHands: 1,
  });
}

async function loadHandDetector() {
  if (detector.loading || detector.ready || detector.failed) {
    return;
  }
  detector.loading = true;
  updateDetectorStatus();

  try {
    const { FilesetResolver, HandLandmarker } = await import(MEDIAPIPE_PACKAGE);
    const vision = await FilesetResolver.forVisionTasks(MEDIAPIPE_WASM);
    try {
      detector.handLandmarker = await createHandLandmarker(
        HandLandmarker,
        vision,
        "GPU",
      );
    } catch {
      detector.handLandmarker = await createHandLandmarker(
        HandLandmarker,
        vision,
        "CPU",
      );
    }
    detector.ready = true;
  } catch {
    detector.failed = true;
  } finally {
    detector.loading = false;
    updateDetectorStatus();
  }
}

function getSourceSize() {
  return {
    width: video.videoWidth || 640,
    height: video.videoHeight || 480,
  };
}

function cropFromLandmarks(landmarks, sourceWidth, sourceHeight) {
  if (!landmarks?.length) {
    return null;
  }

  let minX = sourceWidth;
  let minY = sourceHeight;
  let maxX = 0;
  let maxY = 0;

  for (const landmark of landmarks) {
    const x = Math.min(Math.max(landmark.x * sourceWidth, 0), sourceWidth);
    const y = Math.min(Math.max(landmark.y * sourceHeight, 0), sourceHeight);
    minX = Math.min(minX, x);
    minY = Math.min(minY, y);
    maxX = Math.max(maxX, x);
    maxY = Math.max(maxY, y);
  }

  const handWidth = Math.max(maxX - minX, 1);
  const handHeight = Math.max(maxY - minY, 1);
  const centerX = (minX + maxX) / 2;
  const centerY = (minY + maxY) / 2;
  const maxSquare = Math.min(sourceWidth, sourceHeight);
  let size = Math.max(handWidth, handHeight) * CROP_SCALE + CROP_PADDING;
  size = Math.min(Math.max(size, Math.min(112, maxSquare)), maxSquare);

  let x = centerX - size / 2;
  let y = centerY - size / 2;
  x = Math.min(Math.max(x, 0), sourceWidth - size);
  y = Math.min(Math.max(y, 0), sourceHeight - size);

  return {
    x,
    y,
    width: size,
    height: size,
    sourceWidth,
    sourceHeight,
    mode: "detector",
  };
}

function detectCrop(now, sourceWidth, sourceHeight) {
  if (!detector.ready || !detector.handLandmarker) {
    return null;
  }

  try {
    const result = detector.handLandmarker.detectForVideo(video, now);
    const landmarks = result.landmarks?.[0];
    return cropFromLandmarks(landmarks, sourceWidth, sourceHeight);
  } catch {
    detector.ready = false;
    detector.failed = true;
    updateDetectorStatus();
    return null;
  }
}

function fullFrameCrop(sourceWidth, sourceHeight) {
  return {
    x: 0,
    y: 0,
    width: sourceWidth,
    height: sourceHeight,
    sourceWidth,
    sourceHeight,
    mode: "full",
  };
}

function drawCropToCanvas(crop) {
  captureCanvas.width = modelInputSize;
  captureCanvas.height = modelInputSize;
  captureCtx.drawImage(
    video,
    crop.x,
    crop.y,
    crop.width,
    crop.height,
    0,
    0,
    captureCanvas.width,
    captureCanvas.height,
  );
}

function mapKeypointsToCanvas(keypoints, inputSize, crop) {
  if (!keypoints?.length || !crop) {
    return [];
  }

  const points = keypoints.map(([x, y]) => {
    const sourceX = crop.x + (Number(x) / inputSize) * crop.width;
    const sourceY = crop.y + (Number(y) / inputSize) * crop.height;
    return {
      x: cameraCanvas.width - (sourceX / crop.sourceWidth) * cameraCanvas.width,
      y: (sourceY / crop.sourceHeight) * cameraCanvas.height,
    };
  });
  return scalePoints(points, DISPLAY_KEYPOINT_SCALE);
}

function scalePoints(points, scale) {
  if (!points.length || scale === 1) {
    return points;
  }

  const xs = points.map((point) => point.x);
  const ys = points.map((point) => point.y);
  const centerX = (Math.min(...xs) + Math.max(...xs)) / 2;
  const centerY = (Math.min(...ys) + Math.max(...ys)) / 2;
  return points.map((point) => ({
    x: centerX + (point.x - centerX) * scale,
    y: centerY + (point.y - centerY) * scale,
  }));
}

function canvasToDataUrl(canvas) {
  return canvas.toDataURL("image/jpeg", 0.72);
}

async function sendPrediction(now) {
  if (!cameraReady || predictionBusy || now - lastPredictionAt < 160) {
    return;
  }

  predictionBusy = true;
  lastPredictionAt = now;

  const { width: sourceWidth, height: sourceHeight } = getSourceSize();
  const detectorCrop = detectCrop(now, sourceWidth, sourceHeight);
  const crop = detectorCrop || fullFrameCrop(sourceWidth, sourceHeight);
  latestCrop = crop;
  cropMode.textContent = crop.mode === "detector" ? "crop" : "full frame";
  drawCropToCanvas(crop);

  try {
    const response = await fetch("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        image: canvasToDataUrl(captureCanvas),
        threshold,
      }),
    });
    if (!response.ok) {
      throw new Error(`Prediction failed with HTTP ${response.status}`);
    }

    const prediction = await response.json();
    latestPrediction = prediction;
    latestScore = Number(prediction.openScore || 0);
    latestCanvasKeypoints = mapKeypointsToCanvas(
      prediction.keypoints,
      Number(prediction.inputSize || modelInputSize),
      crop,
    );

    const isOpen = Boolean(prediction.isOpen);
    if (previousOpenState !== null && isOpen !== previousOpenState) {
      jump();
    }
    previousOpenState = isOpen;
  } catch {
    modelStatus.textContent = "Prediction unavailable";
  } finally {
    predictionBusy = false;
  }
}

async function startCamera() {
  const stream = await navigator.mediaDevices.getUserMedia({
    video: {
      width: { ideal: 640 },
      height: { ideal: 480 },
      facingMode: "user",
    },
    audio: false,
  });
  video.srcObject = stream;
  await video.play();
  cameraReady = true;
}

function frame(now) {
  const dt = Math.min((now - lastFrameAt) / 1000, 0.05);
  lastFrameAt = now;
  updateGame(dt);
  drawCamera();
  drawGame();
  updateReadouts();
  sendPrediction(now);
  requestAnimationFrame(frame);
}

cameraButton.addEventListener("click", async () => {
  try {
    await startCamera();
    loadHandDetector();
  } catch {
    modelStatus.textContent = "Camera unavailable";
  }
});

function updateCalibratedThreshold() {
  if (calibratedClosedScore === null || calibratedOpenScore === null) {
    return;
  }
  const nextThreshold = (calibratedClosedScore + calibratedOpenScore) / 2;
  const minThreshold = Number(thresholdSlider.min);
  const maxThreshold = Number(thresholdSlider.max);
  threshold = Math.min(Math.max(nextThreshold, minThreshold), maxThreshold);
  thresholdSlider.value = threshold.toFixed(2);
  updateReadouts();
}

calibrateClosedButton.addEventListener("click", () => {
  if (latestScore > 0) {
    calibratedClosedScore = latestScore;
    updateCalibratedThreshold();
  }
});

calibrateOpenButton.addEventListener("click", () => {
  if (latestScore > 0) {
    calibratedOpenScore = latestScore;
    updateCalibratedThreshold();
  }
});

resetButton.addEventListener("click", resetGame);

thresholdSlider.addEventListener("input", () => {
  threshold = Number(thresholdSlider.value);
  updateReadouts();
});

window.addEventListener("keydown", (event) => {
  if (event.code === "Space" || event.key.toLowerCase() === "w") {
    event.preventDefault();
    jump();
  } else if (event.key.toLowerCase() === "r") {
    resetGame();
  }
});

resetGame();
refreshStatus().catch(() => {
  modelStatus.textContent = "Model status unavailable";
});
loadHandDetector();
requestAnimationFrame(frame);
