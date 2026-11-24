# logistic_model.py
# logistic regression classifier for task 2
# build_lr creates the model, train fits it, evaluate reports metrics
# uses L2 regularisation via the C parameter

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report


def build_lr(C=1.0):
    # logistic regression with L2 regularisation
    # C controls regularisation strength, smaller C = more regularisation
    return LogisticRegression(C=C, max_iter=2000, random_state=42)


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

    model = build_lr(C=1.0)
    model = train(model, X_train, y_train_bin)
    evaluate(model, X_test, y_test_bin)
