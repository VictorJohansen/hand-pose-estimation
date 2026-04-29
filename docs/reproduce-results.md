# Reproduce Results

The default reproduction path uses the committed `webcam-model` `.keras`
checkpoint. Generated evaluation artifacts, result tables, logs, and figures are
ignored by Git. Full retraining is optional.
The three report models are the baseline model, improved model, and webcam
model. The webcam model uses the improved model architecture trained with
additional online augmentation.

From an activated environment, regenerate the validation metrics for the
committed checkpoint:

```bash
python -m src.evaluation.evaluate_run webcam-model
```

Refresh the result table:

```bash
python -m src.evaluation.report_summary webcam-model
```

To recreate the full three-model report comparison, first train or provide local
checkpoints for the baseline model and improved model, then run:

```bash
python -m src.evaluation.evaluate_run baseline-model
python -m src.evaluation.evaluate_run improved-model
python -m src.evaluation.evaluate_run webcam-model
python -m src.evaluation.report_summary baseline-model improved-model webcam-model
python -m src.evaluation.report_figures \
  baseline-model \
  improved-model \
  webcam-model
```

The commands update:

- `artifacts/<run>/evaluation.json`
- `reports/result-summary.md`
- `reports/report-figures/figure1.png` through `figure11.png`

These outputs are generated files and are not committed.

The canonical validation split uses `SPLIT_SEED=42` and
`validation_fraction=0.1`.
