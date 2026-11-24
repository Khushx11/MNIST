# Experimental protocol: reliable digit classification

## Research question

A model can be accurate on conventional test images but become confidently wrong when handwriting changes. This experiment asks how a fixed two-class digit recogniser responds to controlled visual shifts, whether confidence aligns with observed correctness, and whether abstaining on uncertain examples changes the risk/coverage trade-off.

It extends the original Adelaide University Machine Learning Algorithms exercise rather than replacing or retroactively altering its recorded coursework results.

## Fixed design

- **Task:** distinguish handwritten digits 1 and 7 (not full 10-digit MNIST).
- **Source:** the canonical MNIST 60,000-training / 10,000-test partition, loaded from the Keras/TensorFlow-hosted copy.
- **Model:** L2-regularised logistic regression, `C=0.1`, `solver=liblinear`. The hyperparameters are fixed in code, not chosen on the held-out test.
- **Training/validation:** a stratified 80:20 split of the official training examples, random seed 42. The validation subset is never used to fit model weights.
- **Abstention:** select validation-only confidence thresholds at observed risk limits of 0%, 0.1%, 0.5%, 1% and 5%. Report both coverage and conditional accuracy on the validation set and all held-out corruption conditions. A threshold above 1 means no predictions satisfy a given constraint. The 0.5% setting remains the labelled single-threshold reference.
- **Held-out test:** evaluate the unchanged model and unchanged threshold on clean images, Gaussian noise (σ=0.2 in [0,1] pixel units), fixed 20° rotation, and fixed central 8x8 occlusion.
- **Outputs:** accuracy, macro F1, Brier score for digit 7, expected calibration error (10 bins), confusion matrix, selective coverage and selective accuracy per condition.
- **Integrity checks:** audit image-content duplicates across train, validation and test (the script reports any matches, not a presumed zero).

The shifts are deterministic digital transforms, not an external dataset of naturally degraded photographs. Robustness to those transforms does not establish real-world or adversarial robustness.

## Reproduce

```sh
python -m pip install -r requirements.txt
python -m pip install pytest==8.3.2
python -m pytest tests -q
python -m scripts.run_reliability_study --output results/reliability_metrics.json
```

The experiment downloads and caches the ~11 MB canonical MNIST archive at `data/mnist_full.npz`. That file and the generated results directory are ignored by Git. The GitHub Actions `Held-out MNIST reliability study` workflow runs the same evaluation against the official test partition and uploads the measured JSON.

## Caveats for interview interpretation

Expected calibration error depends on the binning definition and sample composition. Probability estimates may become unreliable under shift even if clean-test accuracy remains high. All selective validation risk targets are **observed criteria**, not distribution-free guarantees or conformal prediction statements. A loose target can lead to 100% coverage and no abstention; the full risk-coverage series records that limitation rather than concealing it. The model must still be independently assessed for bias, handwriting styles, deployment latency and out-of-distribution digits.

For external deployment, add evaluation on independently collected handwritten samples rather than treating rotated MNIST as a proxy for all handwriting environments.
