# preprocessor.py
# handles all data preparation before modelling
# flattens images, fits and applies standard scaler, binarises labels
# scaler is always fitted on training data only to avoid data leakage

import numpy as np
from sklearn.preprocessing import StandardScaler


def flatten(X):
    # reshape images from (n, 28, 28) to (n, 784)
    return X.reshape(X.shape[0], -1).astype(float)


def get_scaler(X_train_flat):
    # fit scaler on training data only, never on test data
    scaler = StandardScaler()
    scaler.fit(X_train_flat)
    return scaler


def apply_scaler(scaler, X):
    # apply the already-fitted scaler to any split
    return scaler.transform(X)


def binarise_labels(y):
    # convert original labels (1, 7) to binary (0, 1)
    # digit 1 becomes 0, digit 7 becomes 1
    return (y == 7).astype(int)


if __name__ == '__main__':
    from data_loader import load_data

    X_train_raw, y_train, X_test_raw, y_test = load_data(
        'data/mnist_subsets.npz'
    )

    X_train_flat = flatten(X_train_raw)
    X_test_flat  = flatten(X_test_raw)

    scaler  = get_scaler(X_train_flat)
    X_train = apply_scaler(scaler, X_train_flat)
    X_test  = apply_scaler(scaler, X_test_flat)

    y_train_bin = binarise_labels(y_train)
    y_test_bin  = binarise_labels(y_test)

    print(f'X_train: {X_train.shape}  mean={X_train.mean():.4f}')
    print(f'X_test : {X_test.shape}')
    print(f'y_train unique: {np.unique(y_train_bin)}')
