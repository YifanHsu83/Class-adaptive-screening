# Data preparation

Large data files are not tracked in this repository.

## Public datasets

- Tiny ImageNet: obtain the 200-class dataset from its official distribution.
- CIFAR-10/100: <https://www.cs.toronto.edu/~kriz/cifar.html>
- CIFAR-10N/CIFAR-100N annotations:
  <https://github.com/UCSC-REAL/cifar-10-100n>

Generate frozen, unit-normalized features with:

- OpenAI CLIP ViT-B/32 (512 dimensions), or
- DINOv2 ViT-B/14 (768 dimensions).

Export training and evaluation features as CSV files. Use a common feature
prefix and include integer label columns. The training CSV may additionally
contain a clean reference label for known-noise evaluation. Dataset-specific
column names and paths belong in `configs/datasets.json`.

The software also accepts compressed NPZ files containing `X`, `y`, and
optionally `clean_y` arrays. For NPZ input, set `format` to `npz`; label and
feature-column settings are ignored.
