# Reliable Handwriting Classification: MNIST 1 vs 7

Portfolio by [Khush Bhavin Maru](https://github.com/Khushx11) · Adelaide University MAAIM.

A reproducible machine-learning case study spanning classification, exploratory clustering, calibration, selective prediction, and robustness to distribution shifts. The reliability experiment expands the original coursework with a fully separate train/validation/test protocol.

This project began as individual coursework in **Machine Learning Algorithms at Adelaide University (2026)**. This public repository contains a cleaned source-code and notebook version, separate from the private submission archive.

## My contribution

The original classification and clustering study was my individual Adelaide University coursework. The repository includes the analysis notebook, comparison of classification and clustering methods, and a later reliability extension evaluating calibration, abstention and corruptions on the official held-out MNIST split.

## Advanced reliability extension

The original 100-example coursework study remains available in the notebook. A **separate, larger reliability experiment** now trains on digits 1 and 7 from the official 60,000-example MNIST training partition, selects confidence thresholds at five validation-only risk targets (0%, 0.1%, 0.5%, 1% and 5%), and evaluates unchanged weights and thresholds on the held-out official test partition.

| Capability | Why it matters |
| --- | --- |
| Gaussian noise, rotation and occlusion stress tests | Exposes failures hidden by clean-only accuracy |
| Expected calibration error and Brier score | Checks whether probability confidence tracks actual correctness |
| Selective prediction / abstention | Measures risk and coverage at five validation-selected confidence thresholds, without test-set tuning |
| Exact image duplicate audit | Makes potential cross-partition overlap visible |
| GitHub Actions reproducibility | Executes regression tests and produces measured JSON as a workflow artifact |

```sh
python -m pip install -r requirements.txt
python -m pip install pytest==8.3.2
python -m pytest tests -q
python -m scripts.run_reliability_study --output results/reliability_metrics.json
```

The full reliability study downloads the canonical 60k/10k MNIST split from the Keras/TensorFlow dataset mirror. The runner checks the archive’s SHA-256 on every run and records it in the results JSON. It requires no TensorFlow installation. The study does **not** claim novelty over published handwriting research or guaranteed performance under real-world shift. See [evaluation protocol](docs/reliability_protocol.md) and the latest [reliability workflow](.github/workflows/reliability-evaluation.yml). Reported results should be read from the generated artifact, not invented in advance.

## Verified research results

On **2,163 held-out official MNIST test images** from digits 1 and 7, the reliability extension achieved **99.26% clean accuracy**. Central 8×8 occlusion lowered accuracy to **96.90%** and raised expected calibration error to **0.0702**, compared with **0.0055** on clean test data.

A validation-selected **0% observed-error** abstention cutoff accepted **94.27%** of clean test images, all classified correctly; on rotated images it accepted **83.26%** with **99.83% accuracy among accepted** examples. These numbers were produced by [a successful full-dataset GitHub Actions run](https://github.com/Khushx11/mnist-classification-and-clustering/actions/runs/45383747870). They **do not** guarantee performance on external handwriting or unknown digits.

Read the [complete measured reliability report](docs/measured_reliability.md) for the four stress conditions, coverage/accuracy table and limitations.

## What it covers

- Pixel statistics and PCA-based exploratory analysis
- Logistic regression and an RBF support vector machine
- Stratified cross-validation for hyperparameter selection
- Accuracy, macro F1, confusion matrices and class-level analysis
- K-means clustering, silhouette scores, and centroid visualisation

## Getting started

Python 3.10 or 3.11 is recommended with the pinned historical dependencies.

```bash
git clone https://github.com/Khushx11/mnist-classification-and-clustering.git
cd mnist-classification-and-clustering
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python scripts/prepare_example_data.py
python src/run_tests.py
cd notebooks
jupyter notebook analysis.ipynb
```

The preparation script downloads the MNIST dataset from OpenML and creates a **new deterministic sample** with the same class/count structure as the original coursework: 100 training images and 1,000 test images, restricted to digits 1 and 7. The generated data is ignored by Git.

**Reproduction boundary:** The exact original coursework sample is not distributed in this public repository, so its reported metrics in [the original analysis notes](docs/original-coursework-results.md) may differ from a fresh run of the public example. Do not interpret the original numbers as measurements from this new sample until rerun.

## Repository structure

```text
src/                  Original classification modules plus reliability metrics
notebooks/analysis.ipynb  Clean notebook with executable code cells
scripts/prepare_example_data.py  Recreates original-coursework example subset
scripts/run_reliability_study.py  Runs official-split robustness evaluation
docs/original-coursework-results.md  Original experiment observations
requirements.txt      Reproducible Python dependencies
```

## Dataset attribution

The original MNIST handwritten digits dataset was developed by Yann LeCun, Corinna Cortes and collaborators and is distributed under the Creative Commons Attribution-ShareAlike 3.0 licence. This repository does not include copies of the MNIST images. See [Keras dataset documentation](https://keras.io/api/datasets/mnist/) for dataset provenance and licensing.

## Limitations

Experiments based on 100 training examples are sensitive to sampling choices and class imbalance. Metrics should be interpreted in relation to the specific data split, not as universal performance for all handwritten digits.
