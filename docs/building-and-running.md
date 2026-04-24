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

- Download the [FreiHAND dataset](https://lmb.informatik.uni-freiburg.de/resources/datasets/FreihandDataset.en.html)
- Unzip and place `FreiHAND_pub_v2` in `hand-pose-estimation/data/`
- The default loader path is `data/FreiHAND_pub_v2/`
- `src/data/freihand.py` also accepts a parent path and normalizes it to the dataset root when possible

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

Check that the dataset is visible to the loader:

```bash
python -c "from src.data.freihand import FreiHand; dataset = FreiHand(); dataset.validate(); print(f'Dataset visible: {dataset.root}')"
```

## Notebooks

- Open `notebooks/test_setup.ipynb` first for a quick setup check.
- Use `notebooks/explore_dataset.ipynb` for additional dataset inspection.
