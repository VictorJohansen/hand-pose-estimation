# Training

Retraining is optional for reproducing the report. The committed webcam model
checkpoint is the default path for result regeneration.

Train the baseline model:

```bash
python -m src.training.train_baseline --model-id baseline-model
```

Train the improved model:

```bash
python -m src.training.train_improved --run-name improved-model
```

Train the webcam model. This is the improved model architecture trained with
additional online augmentation:

```bash
python -m src.training.train_improved \
  --run-name webcam-model \
  --online-augmentation
```

Each run writes:

- `models/<run_name>/best.keras`
- `logs/<run_name>/history.json`
- `logs/<run_name>/config.json`

Training uses all FreiHAND RGB variants by default: `gs`, `hom`, `sample`, and
`auto`. Validation uses `gs`. To train on only the original images, pass
`--train-variants gs`; to choose a subset, pass a comma-separated list such as
`--train-variants gs,hom`.

Use `--limit-train` and `--limit-val` for smoke checks. Run the training
commands with `--help` for the full CLI.
