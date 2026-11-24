# MNIST 1-vs-7 Classification and Clustering

A reproducible machine-learning case study comparing logistic regression and radial-basis-function SVM classification, with an additional k-means clustering analysis of handwritten digits.

This project began as individual coursework in **Machine Learning Algorithms at Adelaide University (2026)**. This public repository contains a cleaned source-code and notebook version, separate from the private submission archive.

## What it covers

- Pixel statistics and PCA-based exploratory analysis
- Logistic regression and an RBF support vector machine
- Stratified cross-validation for hyperparameter selection
- Accuracy, macro F1, confusion matrices and class-level analysis
- K-means clustering, silhouette scores, and centroid visualisation

## Getting started

Python 3.10 or 3.11 is recommended with the pinned historical dependencies.

```bash
git clone https://github.com/Khushx11/MNIST.git
cd MNIST
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
src/                  Reusable analysis, model, tuning and plotting modules
notebooks/analysis.ipynb  Clean notebook with executable code cells
scripts/prepare_example_data.py  Recreates an example MNIST subset
docs/original-coursework-results.md  Original experiment observations
requirements.txt      Reproducible Python dependencies
```

## Dataset attribution

The original MNIST handwritten digits dataset was developed by Yann LeCun, Corinna Cortes and collaborators and is distributed under the Creative Commons Attribution-ShareAlike 3.0 licence. This repository does not include copies of the MNIST images. See [Keras dataset documentation](https://keras.io/api/datasets/mnist/) for dataset provenance and licensing.

## Limitations

Experiments based on 100 training examples are sensitive to sampling choices and class imbalance. Metrics should be interpreted in relation to the specific data split, not as universal performance for all handwritten digits.
