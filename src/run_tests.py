# run_tests.py
# quick smoke test to verify all modules load and run correctly
# run this from the mnist_classifier folder before opening the notebook
# usage: python src/run_tests.py

import sys
import numpy as np

sys.path.insert(0, 'src')

print('running module smoke tests...')
print()

# test data_loader
from data_loader import load_data
X_train_raw, y_train, X_test_raw, y_test = load_data(
    'data/mnist_subsets.npz'
)
assert X_train_raw.shape == (100, 28, 28), 'train shape wrong'
assert X_test_raw.shape  == (1000, 28, 28), 'test shape wrong'
assert set(y_train.tolist()) == {1, 7}, 'unexpected labels'
print('data_loader       OK')

# test preprocessor
from preprocessor import flatten, get_scaler, apply_scaler, binarise_labels
X_tr_flat = flatten(X_train_raw)
X_te_flat = flatten(X_test_raw)
scaler    = get_scaler(X_tr_flat)
X_tr      = apply_scaler(scaler, X_tr_flat)
X_te      = apply_scaler(scaler, X_te_flat)
y_tr_bin  = binarise_labels(y_train)
y_te_bin  = binarise_labels(y_test)
assert X_tr.shape == (100, 784), 'flattened train shape wrong'
assert set(y_tr_bin.tolist()) == {0, 1}, 'binarise labels wrong'
print('preprocessor      OK')

# test stats
from stats import class_counts, class_balance_ratio, pixel_stats_per_class
counts = class_counts(y_train)
assert len(counts) == 2, 'should have 2 classes'
ratio  = class_balance_ratio(y_train)
assert 0 < ratio <= 1.0, 'balance ratio out of range'
pstats = pixel_stats_per_class(X_train_raw, y_train)
assert 1 in pstats and 7 in pstats, 'missing class in pixel stats'
print('stats             OK')

# test logistic_model
from logistic_model import build_lr, train, evaluate
lr = build_lr(C=1.0)
lr = train(lr, X_tr, y_tr_bin)
preds_lr, acc_lr, f1_lr = evaluate(lr, X_te, y_te_bin, split_name='smoke')
assert 0.5 < acc_lr <= 1.0, 'LR accuracy looks wrong'
print('logistic_model    OK')

# test svm_model
from svm_model import build_svm, train as svm_train, evaluate as svm_eval
svm = build_svm(C=1.0, gamma=0.001)
svm = svm_train(svm, X_tr, y_tr_bin)
preds_svm, acc_svm, f1_svm = svm_eval(svm, X_te, y_te_bin, split_name='smoke')
assert 0.5 < acc_svm <= 1.0, 'SVM accuracy looks wrong'
print('svm_model         OK')

# test tuner (small range to keep it fast)
from tuner import tune_lr, tune_svm
lr_results, best_C = tune_lr(X_tr, y_tr_bin, C_values=[0.1, 1.0])
assert best_C in [0.1, 1.0], 'best C not in search range'
grid, best_C_s, best_g = tune_svm(
    X_tr, y_tr_bin,
    C_values=[1, 10],
    gamma_values=[0.001, 0.01]
)
assert grid.shape == (2, 2), 'SVM grid shape wrong'
print('tuner             OK')

# test clusterer
from clusterer import (run_kmeans, compute_elbow, compute_silhouette,
                       get_optimal_k, get_cluster_centers_pixels)
X_7  = X_te[y_test == 7]
km5, labels5 = run_kmeans(X_7, n_clusters=5)
assert len(labels5) == X_7.shape[0], 'label count mismatch'
inertias = compute_elbow(X_7, range(2, 5))
assert len(inertias) == 3, 'elbow length wrong'
sil = compute_silhouette(X_7, range(2, 5), sample_size=100)
opt_k, opt_score = get_optimal_k(range(2, 5), sil)
assert 2 <= opt_k <= 4, 'optimal k out of range'
centers = get_cluster_centers_pixels(km5, scaler)
assert centers.shape == (5, 28, 28), 'centre shape wrong'
print('clusterer         OK')

print()
print('all tests passed')
