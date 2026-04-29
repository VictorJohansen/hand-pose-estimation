# Report figure generation

Use `reports/report-figures/` for numbered figures intended for the project report.
Files in that folder are PNG-only and named for the figure number used in the
report text, such as `figure1.png` for Figure 1.

Use `reports/all-figures/` for exploratory notebook output and extra images.

List the configured figures and descriptions:

```bash
python -m src.evaluation.report_figures --list
```

Generate the current numbered report figure set:

```bash
python -m src.evaluation.report_figures \
  baseline-model-1 improved-model-1 improved-model-1-online-augmented
```

Generate only selected figures:

```bash
python -m src.evaluation.report_figures \
  baseline-model-1 improved-model-1 improved-model-1-online-augmented \
  --figures 1 2 8 11
```

Choose the dataset examples shown in `figure2.png`:

```bash
python -m src.evaluation.report_figures \
  baseline-model-1 improved-model-1 improved-model-1-online-augmented \
  --dataset-sample-ids 0 1 2 3
```

Choose the FreiHAND and project-augmentation comparison sample:

```bash
python -m src.evaluation.report_figures \
  baseline-model-1 improved-model-1 improved-model-1-online-augmented \
  --variant-comparison-sample-id 0 \
  --freihand-variant auto \
  --augmentation-seed 42
```

The command writes PNG files to `reports/report-figures/` and refreshes
`docs/report-figure-captions.md`.
