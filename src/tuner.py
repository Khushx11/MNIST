# tuner.py
# cross-validation and hyperparameter search for task 3
# tune_lr searches over C values for logistic regression
# tune_svm runs a grid search over C and gamma for RBF SVM
# all tuning uses stratified k-fold CV on training data only

import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC


def cv_score(model, X, y, n_splits=5):
    # run stratified k-fold cross-validation and return mean macro F1
    # stratified means each fold keeps the same class ratio as the full set
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    scores = cross_val_score(model, X, y, cv=cv, scoring='f1_macro')
    return scores.mean(), scores.std()


def tune_lr(X, y, C_values=None):
    # search over a range of C values for logistic regression
    # returns a list of (C, mean_f1, std_f1) tuples
    if C_values is None:
        C_values = [0.001, 0.01, 0.1, 1, 10, 100, 1000]

    results = []
    for C in C_values:
        model = LogisticRegression(C=C, max_iter=2000, random_state=42)
        mean_f1, std_f1 = cv_score(model, X, y)
        results.append((C, round(mean_f1, 4), round(std_f1, 4)))
        print(f'  C={C:<8}  mean F1={mean_f1:.4f}  std={std_f1:.4f}')

    best = max(results, key=lambda x: x[1])
    print(f'\nBest C for LR: {best[0]}  (CV F1={best[1]})')
    return results, best[0]


def tune_svm(X, y, C_values=None, gamma_values=None):
    # grid search over C and gamma for RBF SVM
    # returns a 2D array of mean F1 scores and the best (C, gamma) pair
    if C_values is None:
        C_values = [0.01, 0.1, 1, 10, 100]
    if gamma_values is None:
        gamma_values = [0.0001, 0.001, 0.01, 0.1, 1]

    grid = np.zeros((len(C_values), len(gamma_values)))

    for i, C in enumerate(C_values):
        for j, gamma in enumerate(gamma_values):
            model = SVC(kernel='rbf', C=C, gamma=gamma, random_state=42)
            mean_f1, _ = cv_score(model, X, y)
            grid[i, j] = round(mean_f1, 4)
            print(f'  C={C:<6} gamma={gamma:<7}  mean F1={mean_f1:.4f}')

    best_idx   = np.unravel_index(np.argmax(grid), grid.shape)
    best_C     = C_values[best_idx[0]]
    best_gamma = gamma_values[best_idx[1]]
    print(f'\nBest SVM params: C={best_C}, gamma={best_gamma}  '
          f'(CV F1={grid[best_idx]:.4f})')
    return grid, best_C, best_gamma


if __name__ == '__main__':
    import sys
    sys.path.insert(0, '.')
    from data_loader import load_data
    from preprocessor import flatten, get_scaler, apply_scaler, binarise_labels

    X_train_raw, y_train, _, _ = load_data('data/mnist_subsets.npz')
    X_flat      = flatten(X_train_raw)
    scaler      = get_scaler(X_flat)
    X_train     = scaler.transform(X_flat)
    y_train_bin = binarise_labels(y_train)

    print('Tuning Logistic Regression...')
    tune_lr(X_train, y_train_bin)

    print('\nTuning SVM...')
    tune_svm(X_train, y_train_bin)
