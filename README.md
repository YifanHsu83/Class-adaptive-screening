# Class-adaptive propensity-score screening

Code for the experiments described in **“Class-Adaptive
Training-Data Screening Using Propensity-Score Distributions in Pretrained
Representation Spaces.”** It starts from fixed CLIP or DINOv2 feature files and
runs the following statistical analyses:

- one-versus-rest assigned-class propensity scores;
- in-sample and five-fold cross-fitted score estimation;
- fixed, Cantelli, Beta, logit-normal, class-empirical, and global-empirical rules;
- matched-random removal;
- 50 matched-seed random-forest evaluations;
- paired tests with Holm adjustment;
- screening-level sensitivity analysis;
- Beta/logit-normal distribution diagnostics;
- known-label-disagreement evaluation for CIFAR-10N and CIFAR-100N.

Raw images and large embedding matrices are intentionally excluded from Git.
See `data/README.md` for expected inputs and official dataset sources.

## Installation

```bash
git clone https://github.com/YifanHsu83/Class-adaptive-screening.git
cd Class-adaptive-screening
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
pytest
```

Python 3.11 or later is required. The package records the experiment settings and
package versions in every output directory. Matching the paper also requires the
same prepared features, labels, splits, and dependency versions; running this
package alone does not establish numerical agreement with the manuscript.

### Windows PowerShell

Run these commands from the repository folder. Calling the virtual environment's
Python directly avoids PowerShell activation-policy changes.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
.\.venv\Scripts\python.exe -m pytest
.\.venv\Scripts\python.exe examples/make_demo_data.py
.\.venv\Scripts\cas-screen.exe extended --config data/prepared/demo/datasets.json --datasets demo --seeds 2 --output outputs/demo
```

The synthetic demo needs no downloaded dataset. It checks the pipeline, not the
paper's findings. Results are saved in `outputs/demo/extended_summary.csv`.

For publishing this repository, see [GitHub upload instructions](GITHUB_UPLOAD.md).

## Input format

Each dataset configuration points to one training CSV and one evaluation CSV.
Feature columns share a prefix such as `clip_` or `dinov2_`. Labels must be
integer-coded. A clean-label column is optional and is used only in the
known-noise experiment.

```text
image_id,recorded_label,clean_label,clip_0001,...,clip_0512
0,3,3,-0.033,...,0.027
1,3,7, 0.012,...,-0.041
```

Copy `configs/datasets.example.json` to `configs/datasets.json` and replace the
placeholder paths. Paths may be absolute or relative to the configuration file.
The local `configs/datasets.json` is ignored by Git. Keep shareable examples in
`configs/datasets.example.json`.

## Reproduce the manuscript analyses

```bash
# Primary 55-class Tiny ImageNet experiment, including in-sample analysis,
# cross-fitting, distribution diagnostics, and sensitivity analysis.
cas-screen primary --config configs/datasets.json --dataset tiny55_clip \
  --output outputs/primary_tiny55_clip

# Extended Table 8 experiment for all configured dataset/encoder settings.
cas-screen extended --config configs/datasets.json \
  --datasets tiny200_clip tiny200_dinov2 cifar10n_clip cifar10n_dinov2 \
             cifar100n_clip cifar100n_dinov2 \
  --output outputs/extended

# CIFAR known-noise evaluation and the four-panel score-distribution figure.
cas-screen known-noise --config configs/datasets.json \
  --datasets cifar10n_clip cifar10n_dinov2 cifar100n_clip cifar100n_dinov2 \
  --output outputs/known_noise
```

For a shorter run, add `--seeds 2` to any command. The primary analysis still runs
distribution bootstraps and the sensitivity grid, so it may take time. The
configured manuscript settings use
50 seeds (`1` through `50`), `gamma=0.20`, five propensity folds, 100 trees, and
maximum tree depth 8. These defaults are stored in `configs/experiment.json`.

## Outputs

Commands write tidy CSV files, figures, and `run_manifest.json`. Important files
include:

- `method_results.csv`: retention and seed-level accuracy/macro-F1;
- `method_summary.csv`: mean performance by screening rule;
- `paired_tests.csv`: paired comparisons against the original baseline;
- `sensitivity_results.csv`: results over screening levels;
- `distribution_diagnostics.csv`: class-specific Beta/logit-normal diagnostics;
- `exploratory_training_f1.csv`: retained-training one-vs-rest F1 values;
- `known_noise_results.csv`: label-disagreement detection results;
- `known_noise_histograms.pdf`: normalized propensity-score distributions.

## Reproducibility notes

1. Screening is estimated from training data only.
2. Evaluation observations are never removed or used to construct cutoffs.
3. Cross-fitting uses stratified folds and out-of-fold predictions.
4. Downstream methods use identical seeds across screening rules.
5. The matched-random baseline removes the same number per class as the
   class-specific empirical rule.
6. Clean CIFAR labels are used only to evaluate known disagreement detection.
7. Scores equal to the cutoff are removed. Tied empirical scores can therefore
   produce retention rates different from exactly `1 - gamma`.

## Code map

- `data.py`: read prepared features and align CSV feature columns.
- `propensity.py`: fit one-versus-rest logistic models and obtain assigned-class scores.
- `cutoffs.py` and `screening.py`: calculate thresholds and select retained observations.
- `evaluation.py` and `statistics.py`: evaluate classifiers and compare matched seeds.
- `diagnostics.py`: check fitted score distributions.
- `experiments.py`: assemble analyses and save results.
- `cli.py`: expose the three experiment commands.

Run `python -m pytest` after changes. The tests check cross-fitting against a
reference implementation, retention counts, input alignment, and seed overrides.

The random-seed paired tests measure stability with respect to model
randomization; they are not a substitute for uncertainty across independently
sampled datasets.

## License

Code is released under the MIT License. Dataset images, noisy-label annotations,
and pretrained model weights remain subject to their original licenses.
