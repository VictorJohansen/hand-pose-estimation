# Building and running

## Setup

Use Python `3.12` and create a project-local `.venv/` from the repository
root.

macOS or Linux:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Windows PowerShell:

```powershell
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Dependencies

- `requirements.txt` installs the default development setup.
- `requirements/base.txt` contains the core runtime and training dependencies.
- `requirements/dev.txt` adds notebook and local exploration tools.

## Dataset

- The default loader path is `data/FreiHAND_pub_v2/`

Install the training dataset directly into the expected location
(`~3.9 GB` download):

macOS or Linux:

```bash
mkdir -p data/FreiHAND_pub_v2 && curl -L https://lmb.informatik.uni-freiburg.de/data/freihand/FreiHAND_pub_v2.zip -o /tmp/FreiHAND_pub_v2.zip && unzip -q /tmp/FreiHAND_pub_v2.zip -d data/FreiHAND_pub_v2 && rm /tmp/FreiHAND_pub_v2.zip
```

Windows PowerShell:

```powershell
New-Item -ItemType Directory -Force data/FreiHAND_pub_v2 | Out-Null
curl.exe -L https://lmb.informatik.uni-freiburg.de/data/freihand/FreiHAND_pub_v2.zip -o $env:TEMP/FreiHAND_pub_v2.zip
Expand-Archive -Path $env:TEMP/FreiHAND_pub_v2.zip -DestinationPath data/FreiHAND_pub_v2 -Force
Remove-Item $env:TEMP/FreiHAND_pub_v2.zip
```

Manual alternative:

- Download the [FreiHAND dataset](https://lmb.informatik.uni-freiburg.de/resources/datasets/FreihandDataset.en.html)
- Unzip it into `hand-pose-estimation/data/FreiHAND_pub_v2/`

Check that the dataset is visible to the loader:

```bash
python -c "from src.data.freihand import FreiHand; dataset = FreiHand(); dataset.validate(); print(f'Dataset visible: {dataset.root}')"
```

### Evaluation set

The FreiHAND authors publicly released ground-truth annotations for the evaluation set. It is used for final test-set evaluation.

Download and unzip into the same `data/FreiHAND_pub_v2/` directory (`~724 MB`):

macOS or Linux:

```bash
curl -L https://lmb.informatik.uni-freiburg.de/data/freihand/FreiHAND_pub_v2_eval.zip -o /tmp/FreiHAND_pub_v2_eval.zip && unzip -qj /tmp/FreiHAND_pub_v2_eval.zip "*/evaluation_xyz.json" -d data/FreiHAND_pub_v2 && rm /tmp/FreiHAND_pub_v2_eval.zip
```

Windows PowerShell:

```powershell
curl.exe -L https://lmb.informatik.uni-freiburg.de/data/freihand/FreiHAND_pub_v2_eval.zip -o $env:TEMP/FreiHAND_pub_v2_eval.zip
Add-Type -AssemblyName System.IO.Compression.FileSystem
$zip = [System.IO.Compression.ZipFile]::OpenRead("$env:TEMP/FreiHAND_pub_v2_eval.zip")
$entry = $zip.Entries | Where-Object { $_.Name -eq "evaluation_xyz.json" }
[System.IO.Compression.ZipFileExtensions]::ExtractToFile($entry, "data/FreiHAND_pub_v2/evaluation_xyz.json", $true)
$zip.Dispose()
Remove-Item $env:TEMP/FreiHAND_pub_v2_eval.zip
```

Manual alternative:

- Download the [FreiHAND dataset](https://lmb.informatik.uni-freiburg.de/resources/datasets/FreihandDataset.en.html)
- Extract only `evaluation_xyz.json` from the zip into `hand-pose-estimation/data/FreiHAND_pub_v2/`

Check that the evaluation set is visible:

```bash
python -c "from src.data.freihand import FreiHand; dataset = FreiHand(split='eval'); dataset.validate(); print('Evaluation set visible')"
```

## Training

Train a baseline CNN from the project root:

```bash
python -m src.training.train_baseline --model-id baseline-model-1
```

Use `--model-id baseline-model-2` for the regularized coordinate-regression baseline.
Train the improved heatmap model with:

```bash
python -m src.training.train_improved
```

Each run writes to:

- `models/<run_name>/best.keras` — best checkpoint by validation loss
- `logs/<run_name>/history.json` — per-epoch training and validation metrics
- `logs/<run_name>/config.json` — run hyperparameters

`<run_name>` defaults to the model ID; override with `--run-name`. Use `--limit-train` and `--limit-val` for fast smoke runs. Run `python -m src.training.train_baseline --help` or `python -m src.training.train_improved --help` for the full CLI.

## Notebooks

- Open `notebooks/explore_dataset.ipynb` for a setup test and dataset visualization.
- Open `notebooks/load_baseline.ipynb` to choose a baseline checkpoint, verify the save/load round-trip, and see prediction overlays on validation samples.
- Open `notebooks/load_improved.ipynb` to choose an improved checkpoint and run the same verification flow.
- Open `notebooks/compare_runs.ipynb` to compare recorded runs.
