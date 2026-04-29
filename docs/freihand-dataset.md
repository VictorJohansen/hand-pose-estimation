# FreiHAND dataset

[Link to FreiHAND dataset](https://lmb.informatik.uni-freiburg.de/resources/datasets/FreihandDataset.en.html)

The training annotations contain `32,560` unique samples.

The folder `training/rgb/` contains `130,240` images. This is because each
unique training sample is provided in `4` RGB versions with different
post-processing. The same keypoint annotations are reused for these image
variants.

Training uses all four variants by default. Validation keeps the original `gs`
variant so validation metrics are measured on the same image view across runs.

The dataset provides `3960` evaluation samples.

## Files we use

In the current project scope, we only use the training images and hand keypoint
annotations.

- `training/rgb/`
  Contains the training RGB images.


- `training_xyz.json`
  Contains the hand keypoint annotations for the training split.
  Each sample has `21` keypoints, and each keypoint has `3` values: `(x, y, z)`.


- `training_K.json`
  Contains the `3 x 3` camera intrinsic matrix for each training sample. This is
  needed to convert the 3D keypoints from `training_xyz.json` into 2D image
  coordinates.

## From 3D to 2D keypoints

This project only uses the keypoints 2D image positions. However, the
coordinates in `training_xyz.json` are not stored as image pixel coordinates.
They are stored as 3D coordinates in camera space. That means we do not get the
correct 2D keypoints by simply dropping the `z` value.

To get 2D keypoints in image space, the 3D keypoints must be projected with
the matching camera matrix from `training_K.json`.
