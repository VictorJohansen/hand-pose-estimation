# Setup

Use Python `3.12` from the repository root. These commands target macOS,
Linux, or WSL2 Ubuntu. Native Windows is not the recommended path for the full
TensorFlow workflow; use WSL2 Ubuntu on Windows.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If `python3.12` is not on `PATH`, install Python 3.12 for your OS and then
rerun the same commands. With `uv`, the fallback is:

```bash
uv python install 3.12
uv python find 3.12
```

`requirements.txt` installs the default local development setup. The dependency
sets are split into `requirements/base.txt` for runtime and training packages
and `requirements/dev.txt` for notebooks and local checks.

## FreiHAND Dataset

Place the FreiHAND dataset under:

```text
data/FreiHAND_pub_v2/
```

Download and extract the main FreiHAND archive:

```bash
mkdir -p data/FreiHAND_pub_v2
curl -L https://lmb.informatik.uni-freiburg.de/data/freihand/FreiHAND_pub_v2.zip \
  -o /tmp/FreiHAND_pub_v2.zip
python - <<'PY'
import zipfile
zipfile.ZipFile('/tmp/FreiHAND_pub_v2.zip').extractall('data/FreiHAND_pub_v2')
PY
rm /tmp/FreiHAND_pub_v2.zip
```

Download and extract the public evaluation annotations into the same directory:

```bash
curl -L https://lmb.informatik.uni-freiburg.de/data/freihand/FreiHAND_pub_v2_eval.zip \
  -o /tmp/FreiHAND_pub_v2_eval.zip
python - <<'PY'
import zipfile

with zipfile.ZipFile('/tmp/FreiHAND_pub_v2_eval.zip') as archive:
    member = next(name for name in archive.namelist() if name.endswith('evaluation_xyz.json'))
    with archive.open(member) as source, open('data/FreiHAND_pub_v2/evaluation_xyz.json', 'wb') as target:
        target.write(source.read())
PY
rm /tmp/FreiHAND_pub_v2_eval.zip
```

The training split has `32,560` unique samples. `training/rgb/` contains
`130,240` images because FreiHAND provides four RGB variants per sample: `gs`,
`hom`, `sample`, and `auto`. Training uses all four variants by default, while
validation uses `gs` only so validation metrics remain comparable across runs.

This project predicts 2D image-space hand keypoints. FreiHAND stores 3D
keypoints in camera space, so the loader projects them to 2D with the matching
camera matrix from `training_K.json` or `evaluation_K.json`.

## Validate The Dataset

Check the training split:

```bash
python -c "from src.data.freihand import FreiHand; dataset = FreiHand(); dataset.validate(); print(f'Dataset visible: {dataset.root}')"
```

Check the evaluation split:

```bash
python -c "from src.data.freihand import FreiHand; dataset = FreiHand(split='eval'); dataset.validate(); print('Evaluation set visible')"
```
