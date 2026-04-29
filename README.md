# hand-pose-estimation

DAT255 semester project for 2D hand keypoint estimation on the FreiHAND
dataset. The repository contains the code, the selected webcam model
checkpoint, notebooks, and tooling needed to check results locally.

## Reproduction Path

Follow these in order from the repository root:

1. [Set up Python and FreiHAND](docs/setup.md)
2. [Reproduce the reported results from checkpoints](docs/reproduce-results.md)
3. Optionally [retrain the models](docs/training.md)
4. Optionally walk through the notebooks:
   - `notebooks/01_dataset_check.ipynb`
   - `notebooks/02_checkpoint_verification.ipynb`
   - `notebooks/03_report_results.ipynb`

Generated result tables and report figures are intentionally ignored by Git.
They are written to `reports/result-summary.md` and `reports/report-figures/`
when regenerated.

## Quick Commands

Running these commands regenerates result artifacts from the committed webcam
model checkpoint; it does not retrain the models.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

python -m src.evaluation.evaluate_run webcam-model
python -m src.evaluation.report_summary webcam-model
```

## Web Deployment

Web deployment is future work. The expected path is a small web app that loads a
trained `.keras` checkpoint and runs image or webcam-frame inference. It is not
required to reproduce the report results in this repository.
