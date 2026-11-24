# stats.py
# computes descriptive statistics for task 1
# class counts, balance ratio, per-class pixel mean/std/min/max/nonzero

import numpy as np


def class_counts(y):
    # count how many samples belong to each class
    labels, counts = np.unique(y, return_counts=True)
    return dict(zip(labels.tolist(), counts.tolist()))


def class_balance_ratio(y):
    # ratio of minority to majority class, 1.0 means perfectly balanced
    counts = list(class_counts(y).values())
    return round(min(counts) / max(counts), 4)


def pixel_stats_per_class(X_raw, y):
    # X_raw shape: (n, 28, 28), values in [0, 255]
    # returns a dict with per-class pixel mean, std, min, max,
    # and percentage of non-zero pixels
    results = {}
    for label in np.unique(y):
        imgs = X_raw[y == label].astype(float)
        flat = imgs.flatten()
        results[int(label)] = {
            'n_samples'     : len(imgs),
            'pixel_mean'    : round(float(flat.mean()), 4),
            'pixel_std'     : round(float(flat.std()), 4),
            'pixel_min'     : int(flat.min()),
            'pixel_max'     : int(flat.max()),
            'nonzero_pct'   : round(float((flat > 0).mean() * 100), 2),
        }
    return results


def print_summary(X_raw, y, split_name='Train'):
    print(f'--- {split_name} set summary ---')
    print(f'Total samples : {len(y)}')
    print(f'Class counts  : {class_counts(y)}')
    print(f'Balance ratio : {class_balance_ratio(y)}')
    print()
    for label, stats in pixel_stats_per_class(X_raw, y).items():
        print(f'Digit {label}:')
        for k, v in stats.items():
            print(f'  {k:<15}: {v}')
        print()


if __name__ == '__main__':
    import sys
    sys.path.insert(0, '.')
    from data_loader import load_data

    X_train, y_train, X_test, y_test = load_data(
        'data/mnist_subsets.npz'
    )
    print_summary(X_train, y_train, split_name='Train')
