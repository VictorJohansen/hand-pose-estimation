# Report figure generation

Use `reports/report-figures/` for figures intended for the project report.
Use `reports/all-figures/` for exploratory notebook output and extra images.

Generate the current report figure set:

```bash
python -m src.evaluation.report_figures \
  baseline-model-1 improved-model-1 improved-model-1-online-augmented
```

Generate to an explicit output folder:

```bash
python -m src.evaluation.report_figures \
  baseline-model-1 improved-model-1 improved-model-1-online-augmented \
  --output-dir reports/report-figures
```

Choose the dataset examples shown in `report_dataset_examples`:

```bash
python -m src.evaluation.report_figures \
  baseline-model-1 improved-model-1 improved-model-1-online-augmented \
  --dataset-samples 15550 28457 18199 6097
```

Skip prediction overlays when only dataset and metric figures are needed:

```bash
python -m src.evaluation.report_figures \
  baseline-model-1 improved-model-1 improved-model-1-online-augmented \
  --skip-predictions
```

The command writes PNG and PDF versions for each generated figure.
