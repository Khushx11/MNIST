# svm_model.py
# kernel SVM with RBF kernel for task 3
# non-linear classifier used as the second model in the comparison
# C controls margin penalty, gamma controls influence radius per sample

import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, classification_report


def build_svm(C=1.0, gamma=0.001):
    # RBF kernel SVM, a non-linear classifier
    # C controls the penalty for misclassified points
    # gamma controls how far the influence of a single training sample reaches
    # small gamma = broad influence, large gamma = very local influence
    return SVC(kernel='rbf', C=C, gamma=gamma, random_state=42)


def train(model, X_train, y_train):
    model.fit(X_train, y_train)
    return model


def evaluate(model, X, y, split_name='Test'):
    preds = model.predict(X)
    acc   = accuracy_score(y, preds)
    f1    = f1_score(y, preds, average='macro')

    print(f'--- {split_name} set results ---')
    print(f'Accuracy  : {acc:.4f}')
    print(f'Macro F1  : {f1:.4f}')
    print()
    print(classification_report(y, preds, target_names=['Digit 1', 'Digit 7']))
    return preds, acc, f1


if __name__ == '__main__':
    import sys
    sys.path.insert(0, '.')
    from data_loader import load_data
    from preprocessor import flatten, get_scaler, apply_scaler, binarise_labels

    X_train_raw, y_train, X_test_raw, y_test = load_data(
        'data/mnist_subsets.npz'
    )

    X_train_flat = flatten(X_train_raw)
    X_test_flat  = flatten(X_test_raw)
    scaler       = get_scaler(X_train_flat)
    X_train      = apply_scaler(scaler, X_train_flat)
    X_test       = apply_scaler(scaler, X_test_flat)
    y_train_bin  = binarise_labels(y_train)
    y_test_bin   = binarise_labels(y_test)

    model = build_svm(C=1.0, gamma=0.001)
    model = train(model, X_train, y_train_bin)
    evaluate(model, X_test, y_test_bin)
