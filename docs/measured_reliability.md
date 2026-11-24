# Measured robustness and abstention results

These numbers were produced by the successful [full MNIST reliability workflow](https://github.com/Khushx11/mnist-classification-and-clustering/actions/runs/45383747870) on the official MNIST digits 1 and 7. The experiment used **10,405 training**, **2,602 validation** and **2,163 untouched official test** images. Logistic regression hyperparameters were fixed; confidence thresholds were selected *only* on validation data and reused without modification on all test conditions.

## Plain classification and confidence quality

| Test condition | All-example accuracy | Macro F1 | Expected calibration error |
|---|---:|---:|---:|
| Clean official test | 99.26% | 0.9926 | 0.0055 |
| Gaussian noise (σ=0.2) | 99.45% | 0.9944 | 0.0186 |
| 20° rotation | 98.24% | 0.9824 | 0.0113 |
| Central 8×8 occlusion | 96.90% | 0.9690 | 0.0702 |

The modest noise perturbation improved classification accuracy in this specific experiment; it did **not** improve calibration. This illustrates why accuracy and calibration need separate reporting and why robustness should not be presumed monotonic in corruption severity.

## Selective prediction measured on the held-out test

**0% observed validation error target:** Choose the confidence cutoff on validation such that *none of the accepted validation examples* are wrong. This is an empirical validation criterion, not a guarantee of 100% accuracy on future images.

| Test condition | Accepted coverage | Accuracy among accepted | Rejected images |
|---|---:|---:|---:|
| Clean | 94.27% | 100.00% | 124 / 2,163 |
| Gaussian noise | 80.77% | 100.00% | 416 / 2,163 |
| 20° rotation | 83.26% | 99.83% | 362 / 2,163 |
| Central occlusion | 47.20% | 100.00% | 1,142 / 2,163 |

At **0.1% allowed validation error**, the clean-test coverage increased to **97.60%**, with **99.86% accuracy among accepted** examples. At 0.5%, the validation-selected cutoff accepted all examples and therefore provided no abstention benefit in the tested conditions. The complete reproducible JSON also includes curves for 1% and 5% targets.

### Interpretation

The 0% validation target did not ensure zero test error: the 20° rotation subset retained a small number of wrong accepted predictions. Under occlusion, maintaining high selective accuracy required rejecting over half the images. This tradeoff is more informative than reporting clean accuracy alone.

These measurements are restricted to the two digit classes and synthetic corruptions. A real application needs an unknown-digit rejection test, external handwriting styles and independently gathered images. The experiment is a demonstration of model reliability analysis, not a claim of state-of-the-art handwriting recognition.

[Full measured JSON artifact in GitHub Actions](https://github.com/Khushx11/mnist-classification-and-clustering/actions/runs/45383747870) · [Experimental protocol](reliability_protocol.md)
