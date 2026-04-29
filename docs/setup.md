# Setup

Use Python `3.12` from the repository root.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

`requirements.txt` installs the default local development setup. The dependency
sets are split into `requirements/base.txt` for runtime and training packages
and `requirements/dev.txt` for notebooks and local checks.

## FreiHAND Dataset

Place the FreiHAND dataset under:

```text
data/FreiHAND_pub_v2/
```

Download and extract the training dataset:

```bash
mkdir -p data/FreiHAND_pub_v2
curl -L https://lmb.informatik.uni-freiburg.de/data/freihand/FreiHAND_pub_v2.zip \
  -o /tmp/FreiHAND_pub_v2.zip
unzip -q /tmp/FreiHAND_pub_v2.zip -d data/FreiHAND_pub_v2
rm /tmp/FreiHAND_pub_v2.zip
```

Download and extract the public evaluation annotations into the same directory:

```bash
curl -L https://lmb.informatik.uni-freiburg.de/data/freihand/FreiHAND_pub_v2_eval.zip \
  -o /tmp/FreiHAND_pub_v2_eval.zip
unzip -qj /tmp/FreiHAND_pub_v2_eval.zip "*/evaluation_xyz.json" \
  -d data/FreiHAND_pub_v2
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
