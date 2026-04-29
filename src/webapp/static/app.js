const video = document.getElementById("video");
const cameraCanvas = document.getElementById("cameraCanvas");
const cameraCtx = cameraCanvas.getContext("2d");
const gameCanvas = document.getElementById("gameCanvas");
const gameCtx = gameCanvas.getContext("2d");

const cameraButton = document.getElementById("cameraButton");
const calibrateButton = document.getElementById("calibrateButton");
const resetButton = document.getElementById("resetButton");
const modelStatus = document.getElementById("modelStatus");
const gestureState = document.getElementById("gestureState");
const thresholdSlider = document.getElementById("thresholdSlider");
const thresholdValue = document.getElementById("thresholdValue");
const scoreValue = document.getElementById("scoreValue");
const openScoreValue = document.getElementById("openScoreValue");

const captureCanvas = document.createElement("canvas");
captureCanvas.width = 224;
captureCanvas.height = 224;
const captureCtx = captureCanvas.getContext("2d");

const HAND_CONNECTIONS = [
  [0, 1], [1, 2], [2, 3], [3, 4],
  [0, 5], [5, 6], [6, 7], [7, 8],
  [0, 9], [9, 10], [10, 11], [11, 12],
  [0, 13], [13, 14], [14, 15], [15, 16],
  [0, 17], [17, 18], [18, 19], [19, 20],
];

let cameraReady = false;
let modelLoaded = false;
let latestPrediction = null;
let latestScore = 0;
let threshold = Number(thresholdSlider.value);
let wasOpen = false;
let predictionBusy = false;
let lastPredictionAt = 0;
let lastFrameAt = performance.now();

const game = {
  groundY: 302,
  dinoX: 76,
  dinoY: 248,
  dinoWidth: 42,
  dinoHeight: 54,
  velocityY: 0,
  speed: 285,
  score: 0,
  spawnTimer: 0,
  obstacles: [],
  gameOver: false,
};

function resetGame() {
  game.dinoY = game.groundY - game.dinoHeight;
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
  if (game.dinoY >= game.groundY - game.dinoHeight - 0.1) {
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
  game.dinoY += game.velocityY * dt;
  const groundTop = game.groundY - game.dinoHeight;
  if (game.dinoY > groundTop) {
    game.dinoY = groundTop;
    game.velocityY = 0;
  }

  game.spawnTimer -= dt;
  if (game.spawnTimer <= 0) {
    game.obstacles.push({
      x: gameCanvas.width + 20,
      width: 26 + Math.floor(Math.random() * 18),
      height: 38 + Math.floor(Math.random() * 34),
    });
    game.spawnTimer = 0.9 + Math.random() * 0.65;
  }

  for (const obstacle of game.obstacles) {
    obstacle.x -= game.speed * dt;
  }
  game.obstacles = game.obstacles.filter((obstacle) => obstacle.x + obstacle.width > 0);

  if (collides()) {
    game.gameOver = true;
  }
}

function collides() {
  const dinoLeft = game.dinoX;
  const dinoRight = game.dinoX + game.dinoWidth;
  const dinoTop = game.dinoY;
  const dinoBottom = game.dinoY + game.dinoHeight;

  return game.obstacles.some((obstacle) => {
    const left = obstacle.x;
    const right = obstacle.x + obstacle.width;
    const top = game.groundY - obstacle.height;
    const bottom = game.groundY;
    return dinoRight > left && dinoLeft < right && dinoBottom > top && dinoTop < bottom;
  });
}

function drawGame() {
  gameCtx.clearRect(0, 0, gameCanvas.width, gameCanvas.height);
  gameCtx.fillStyle = "#f9faf7";
  gameCtx.fillRect(0, 0, gameCanvas.width, gameCanvas.height);

  gameCtx.strokeStyle = "#5b635f";
  gameCtx.lineWidth = 3;
  gameCtx.beginPath();
  gameCtx.moveTo(0, game.groundY);
  gameCtx.lineTo(gameCanvas.width, game.groundY);
  gameCtx.stroke();

  gameCtx.fillStyle = "#2f8f5b";
  gameCtx.fillRect(game.dinoX, game.dinoY, game.dinoWidth, game.dinoHeight);
  gameCtx.fillStyle = "#111513";
  gameCtx.beginPath();
  gameCtx.arc(game.dinoX + 31, game.dinoY + 13, 3, 0, Math.PI * 2);
  gameCtx.fill();
  gameCtx.fillStyle = "#2f8f5b";
  gameCtx.fillRect(game.dinoX + 7, game.dinoY + game.dinoHeight - 7, 10, 12);
  gameCtx.fillRect(game.dinoX + 27, game.dinoY + game.dinoHeight - 7, 10, 12);

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
    gameCtx.fillStyle = "rgba(255, 255, 255, 0.9)";
    gameCtx.fillRect(216, 120, 328, 98);
    gameCtx.strokeStyle = "#5b635f";
    gameCtx.strokeRect(216, 120, 328, 98);
    gameCtx.fillStyle = "#1d2522";
    gameCtx.font = "32px Arial";
    gameCtx.fillText("GAME OVER", 278, 160);
    gameCtx.font = "18px Arial";
    gameCtx.fillText("reset or open hand", 292, 196);
  }
}

function drawCamera() {
  cameraCtx.fillStyle = "#101613";
  cameraCtx.fillRect(0, 0, cameraCanvas.width, cameraCanvas.height);

  if (cameraReady) {
    cameraCtx.save();
    cameraCtx.translate(cameraCanvas.width, 0);
    cameraCtx.scale(-1, 1);
    cameraCtx.drawImage(video, 0, 0, cameraCanvas.width, cameraCanvas.height);
    cameraCtx.restore();
  }

  if (latestPrediction?.keypoints?.length) {
    drawKeypoints(latestPrediction.keypoints, latestPrediction.inputSize || 224);
  }
}

function drawKeypoints(keypoints, inputSize) {
  const sx = cameraCanvas.width / inputSize;
  const sy = cameraCanvas.height / inputSize;
  cameraCtx.lineWidth = 2;
  cameraCtx.strokeStyle = "#18e27b";
  for (const [start, end] of HAND_CONNECTIONS) {
    cameraCtx.beginPath();
    cameraCtx.moveTo(keypoints[start][0] * sx, keypoints[start][1] * sy);
    cameraCtx.lineTo(keypoints[end][0] * sx, keypoints[end][1] * sy);
    cameraCtx.stroke();
  }
  cameraCtx.fillStyle = "#ff3f62";
  for (const point of keypoints) {
    cameraCtx.beginPath();
    cameraCtx.arc(point[0] * sx, point[1] * sy, 4, 0, Math.PI * 2);
    cameraCtx.fill();
  }
}

function updateReadouts() {
  scoreValue.textContent = String(Math.floor(game.score)).padStart(4, "0");
  openScoreValue.textContent = latestScore.toFixed(2);
  thresholdValue.textContent = threshold.toFixed(2);
  const isOpen = Boolean(latestPrediction?.isOpen);
  gestureState.textContent = isOpen ? "open" : "closed";
  gestureState.classList.toggle("open", isOpen);
}

async function refreshStatus() {
  const response = await fetch("/api/status");
  const status = await response.json();
  modelLoaded = Boolean(status.modelLoaded);
  const modelName = status.modelPath || "no checkpoint";
  const note = status.message ? ` | ${status.message}` : "";
  modelStatus.textContent = `${modelLoaded ? "Model loaded" : "Model missing"}: ${modelName}${note}`;
  if (status.threshold) {
    threshold = Number(status.threshold);
    thresholdSlider.value = threshold.toFixed(2);
  }
  updateReadouts();
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

  captureCtx.save();
  captureCtx.translate(captureCanvas.width, 0);
  captureCtx.scale(-1, 1);
  captureCtx.drawImage(video, 0, 0, captureCanvas.width, captureCanvas.height);
  captureCtx.restore();
  const image = canvasToDataUrl(captureCanvas);

  try {
    const response = await fetch("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ image, threshold }),
    });
    const prediction = await response.json();
    latestPrediction = prediction;
    latestScore = Number(prediction.openScore || 0);
    const isOpen = Boolean(prediction.isOpen);
    if (isOpen && !wasOpen) {
      jump();
    }
    wasOpen = isOpen;
  } catch (error) {
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
  } catch (error) {
    modelStatus.textContent = "Camera unavailable";
  }
});

calibrateButton.addEventListener("click", () => {
  if (latestScore > 0) {
    threshold = Math.max(latestScore * 0.9, 0.1);
    thresholdSlider.value = threshold.toFixed(2);
    updateReadouts();
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
requestAnimationFrame(frame);
