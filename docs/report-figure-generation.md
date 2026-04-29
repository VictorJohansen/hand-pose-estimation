# Report figure generation

Use `reports/figures/` for numbered figures intended for the project report.
Files in that folder are PNG-only and named `figureX.png`, where `X` is the
figure number used in the report text.

Use `reports/all-figures/` for exploratory notebook output and extra images.

List the configured figures and descriptions:

```bash
python -m src.evaluation.report_figure_set --list
```

Generate the current numbered report figure set:

```bash
python -m src.evaluation.report_figure_set
```

Generate only selected figures:

```bash
python -m src.evaluation.report_figure_set 1 3 7
```

Generate with an explicit run set:

```bash
python -m src.evaluation.report_figure_set \
  --runs baseline-model-1 improved-model-1 improved-model-1-online-augmented
```

Choose the dataset examples shown in `figure1.png`:

```bash
python -m src.evaluation.report_figure_set \
  --dataset-samples 15550 28457 18199 6097
```

The command writes PNG files to `reports/figures/` and refreshes
`docs/report-figure-captions.md`.
