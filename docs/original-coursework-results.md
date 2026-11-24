# Original coursework experiment record

The exact assignment dataset is **not included** in this public repository. Results below were obtained on the original private assignment split and may not be reproduced by the provided example-data generator.

# MNIST Binary Classification: Digits 1 vs 7

**Author:** Khush Bhavin Maru
**Course:** Machine Learning Algorithms, Adelaide University (2026)

Binary classification and clustering analysis on a subset of the
MNIST dataset. The dataset contains only digits 1 and 7, with 100
labeled training images and 1000 labeled test images.

---

## Project structure

    mnist_classifier/
    ├── data/                        put the .npz data file here
    ├── notebooks/
    │   └── analysis.ipynb           main notebook, run this
    ├── src/
    │   ├── data_loader.py           loads train/test arrays from .npz
    │   ├── preprocessor.py          flatten, scale, binarise labels
    │   ├── stats.py                 descriptive statistics for task 1
    │   ├── visualiser.py            plotting helpers for task 1
    │   ├── logistic_model.py        logistic regression for task 2
    │   ├── svm_model.py             kernel SVM RBF for task 3
    │   ├── tuner.py                 cross-validation and hyperparam search
    │   └── clusterer.py             k-means clustering for task 4
    ├── outputs/                     saved figures written here by notebook
    ├── requirements.txt
    └── README.md

---

## How to run

1. Clone the repo and go into the project folder
2. Install dependencies:

        pip install -r requirements.txt

3. The original coursework dataset is currently included at:

        data/mnist_subsets.npz

   Confirm the dataset's redistribution terms before making the repository public.

4. Open and run the notebook:

        jupyter notebook notebooks/analysis.ipynb

5. Run all cells from top to bottom in order.
   Output figures are saved to the outputs folder automatically.

---

## Tasks covered

### Task 1: Data Exploration and Analysis
- Loads and inspects the dataset structure
- Reports class counts, balance ratio, pixel mean, std, min, max,
  and non-zero pixel percentage per class
- Four visualisations: sample images, per-class pixel stats,
  class distribution bar chart, PCA scatter plot
- Preprocessing: flatten to 784-d vectors, StandardScaler fitted
  on training data only, labels binarised to 0 and 1

### Task 2: Build a Classification System
- Model: Logistic Regression with L2 regularisation
- Justified by small dataset size and linear separability of the
  two digit classes in pixel space
- Metrics: accuracy and macro F1 score
- Evaluated on test set with confusion matrix and weight visualisation

### Task 3: Comparison of Two Classifiers and Hyperparameter Tuning
- Second model: Kernel SVM with RBF kernel (non-linear)
- Logistic Regression: C tuned over 7 values via 5-fold stratified CV
- SVM: C and gamma tuned over a 5x5 grid via 5-fold stratified CV
- Best hyperparameters selected by macro F1 on validation folds
- Per-class accuracy breakdown included for both models
- Test set only used for final evaluation after tuning is complete

### Task 4: Clustering Analysis of the Test Set
- Digit 7 samples extracted from the test set
- K-means run with k=100 as specified
- Optimal k found using elbow method and silhouette score over k=2 to 40
- Cluster centres visualised as 28x28 images
- Most representative real sample shown per cluster

---

## Results and benchmarks

All results are from the 1000-sample test set. The test set 
was never used during training or hyperparameter tuning.

### Task 2: Logistic Regression (C=1.0, default)

| Metric | Value |
|--------|-------|
| Test accuracy | 0.9470 |
| Test macro F1 | 0.8776 |
| Digit 1 precision / recall | 0.66 / 0.97 |
| Digit 7 precision / recall | 1.00 / 0.94 |

### Task 3: Final model comparison (best CV hyperparameters)

| Model | Best params | Test accuracy | Test macro F1 |
|-------|-------------|---------------|---------------|
| Logistic Regression | C=0.1 | 0.9460 | 0.8740 |
| Kernel SVM RBF | C=100, gamma=0.0001 | 0.9480 | 0.8723 |

Per-class accuracy breakdown:

| Model | Digit 1 acc | Digit 7 acc | Overall acc |
|-------|-------------|-------------|-------------|
| Logistic Regression | 0.9500 | 0.9456 | 0.9460 |
| Kernel SVM RBF | 0.8900 | 0.9544 | 0.9480 |

### Task 4: Clustering (digit 7 test samples)

| Metric | Value |
|--------|-------|
| Digit 7 test samples | 900 |
| K-means k=100 inertia | 811,351.79 |
| Optimal k (silhouette) | 2 |
| Silhouette score at k=2 | 0.8151 |
| Cluster 1 size | 886 samples |
| Cluster 2 size | 14 samples |

Both clusters capture real handwriting style variation: 
standard digit 7 (no crossbar) vs European crossbar digit 7.

## GitHub

https://github.com/Khushx11/MNIST-Assignment

---

## Output figures

Running the notebook produces these figures in the outputs folder:

| File | Description |
|------|-------------|
| viz1_sample_images.png | 8 sample images per class |
| viz2_pixel_stats.png | Mean image, std map, intensity histogram |
| viz3_class_distribution.png | Class counts in train and test splits |
| viz4_pca.png | PCA scatter plot of training data |
| viz4_lr_confusion.png | Confusion matrix for default LR |
| viz5_lr_cv.png | LR cross-validation curve over C values |
| viz6_svm_cv_heatmap.png | SVM grid search heatmap C x gamma |
| viz7_final_comparison.png | Side by side confusion matrices |
| viz8_elbow_silhouette.png | Elbow and silhouette plots for clustering |
| viz9_cluster_centres.png | All cluster centres as 28x28 images |
| viz10_representative_samples.png | Closest real sample per cluster |
| viz_lr_weights.png | LR weight vector reshaped to 28x28 |

---

## Requirements

    numpy==1.26.4
    matplotlib==3.8.4
    seaborn==0.13.2
    scikit-learn==1.4.2
    jupyter==1.0.0
    notebook==7.1.3
    ipykernel==6.29.4

---

## Verification

To verify the full submission runs correctly:

1. Place the data file at data/mnist_subsets.npz
2. Run the smoke tests first:

        python src/run_tests.py

   Expected output: all tests passed

3. Open and run the notebook:

        jupyter notebook notebooks/analysis.ipynb

4. Check the outputs folder for all 12 saved figures listed above.

All code is in the src folder. The notebook imports from src and
calls those functions. No internet access or external APIs needed.

---

**Author:** Khush Bhavin Maru
**Course:** Machine Learning Algorithms, Adelaide University (2026)
