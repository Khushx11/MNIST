# data_loader.py
# loads the mnist 1 vs 7 subset from the .npz file
# returns raw numpy arrays for train and test splits
# used as the entry point for all data access in this project

import numpy as np


def load_data(filepath):
    """
    Load the MNIST 1 vs 7 subset from a .npz file.

    Parameters
    ----------
    filepath : str
        Path to the .npz data file.

    Returns
    -------
    X_train : np.ndarray, shape (100, 28, 28)
    y_train : np.ndarray, shape (100,)
    X_test  : np.ndarray, shape (1000, 28, 28)
    y_test  : np.ndarray, shape (1000,)
    """
    data = np.load(filepath)
    X_train = data['train_data']
    y_train = data['train_targets']
    X_test  = data['test_data']
    y_test  = data['test_targets']
    return X_train, y_train, X_test, y_test


if __name__ == '__main__':
    X_train, y_train, X_test, y_test = load_data('data/mnist_subsets.npz')
    print(f'Train: {X_train.shape}, labels: {y_train.shape}')
    print(f'Test : {X_test.shape},  labels: {y_test.shape}')
    print(f'Unique labels: {sorted(set(y_train))}')
