# Artifact and output locations

Generated outputs should use the following top-level directories so training,
evaluation, and report generation stay reproducible.

## Directory conventions

- `models/`
  Saved trained model checkpoints and exported models.

- `logs/`
  Training logs, TensorBoard event files, and run histories.

- `reports/figures/`
  Report-ready figures such as training curves, prediction overlays, and
  comparison plots. Regenerate the report figure set with
  `python -m src.evaluation.report_figures`.

- `reports/result-summary.md`
  Generated result tables for evaluated runs. Regenerate it with
  `python -m src.evaluation.report_summary`.

- `artifacts/`
  Evaluation outputs that are not final report figures, such as prediction
  CSV/JSON files, metric summaries, generated tables, and intermediate analysis
  files.

## Git tracking

The directories are kept in Git with `.gitkeep` files, but generated contents
inside them are ignored. Commit only small, intentional artifacts needed to
understand or reproduce results in the final report.

Intentional report figures use the `reports/figures/report_*` prefix and are
allowed through `.gitignore` so they can be included with the report source.
