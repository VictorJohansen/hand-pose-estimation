# Building and running

## Python

Use Python `3.12`.

## Environment setup

Create a project-local `.venv/` from the repository root.

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

## Dependency files

- `requirements.txt` installs the default development setup.
- `requirements/base.txt` contains the core runtime and training dependencies.
- `requirements/dev.txt` adds notebook and local exploration tools.

## Dataset

- Download the [FreiHAND dataset](https://lmb.informatik.uni-freiburg.de/resources/datasets/FreihandDataset.en.html)
- Unzip and place `FreiHAND_pub_v2` in `hand-pose-estimation/data/`
