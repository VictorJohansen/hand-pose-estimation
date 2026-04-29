# Webcam T-Rex Game Demo

This demo connects the hand-pose model to a small T-Rex-style runner game.
The player jumps by opening their hand in front of the webcam. The model
predicts 21 hand keypoints; the demo converts those keypoints into a
scale-normalized hand-openness score.

To test the game immediately, before the final model checkpoint exists:

```powershell
venv\Scripts\python.exe -m src.demo.trex_webcam --keyboard-only
```

To test the full model-loading path with the current smoke checkpoint:

```powershell
venv\Scripts\python.exe -m src.demo.trex_webcam --model models\improved-aug-smoke\best.keras
```

For the final demo, run from the project root with the trained checkpoint:

```powershell
venv\Scripts\python.exe -m src.demo.trex_webcam --model models\improved-model-2\best.keras
```

If you restore the earlier reported checkpoint instead, use:

```powershell
venv\Scripts\python.exe -m src.demo.trex_webcam --model models\improved-model-1\best.keras
```

Without `--model`, the script tries known project checkpoints automatically.
If no checkpoint is available, it starts in keyboard fallback mode so the game
can still be tested.

If `--model` points to a checkpoint that does not exist yet, the script prints
a warning and falls back to the best available checkpoint. Add
`--strict-model-path` if you want missing checkpoint paths to fail immediately.

The `improved-aug-smoke` checkpoint is only a wiring test. For the final demo,
use a real trained checkpoint such as `models/improved-model-1/best.keras` or
`models/improved-model-2/best.keras`.

## Controls

- Open hand: jump
- `Space`: manual jump fallback
- `C`: calibrate the current open-hand score as the jump threshold
- `[` / `]`: lower or raise the threshold
- `R`: reset after game over
- `Q` or `Esc`: quit

## Gesture Rule

Before presenting the demo, hold an open hand in front of the camera and press
`C`. This sets the jump threshold from the current model predictions, which is
useful when swapping between partially trained and final checkpoints.

The demo uses the FreiHAND/MANO keypoint order:

- `0`: wrist
- `4, 8, 12, 16, 20`: fingertips
- `5, 9, 13, 17`: palm reference joints

The openness score combines:

1. fingertip distance from the wrist
2. spread between adjacent fingertips
3. normalization by palm size

This makes the control rule simple enough to explain in the report while still
being connected directly to the model output.

## Report Wording

Suggested wording:

> To demonstrate the practical use of the hand-pose estimator, we connected
> the trained model to a webcam-controlled T-Rex runner game. Each webcam frame
> is passed through the model, producing 21 predicted hand keypoints. A simple
> gesture classifier computes a normalized hand-openness score from fingertip
> distances and fingertip spread. When the score crosses a calibrated threshold,
> the dinosaur jumps. This shows how the keypoint model can be used as an
> interaction layer, not only as an offline prediction model.
