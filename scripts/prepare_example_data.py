"""Download MNIST and create a deterministic, independent illustrative subset.

This is NOT the original assignment-specific sample. Requires network access.
"""
from pathlib import Path
import argparse
import numpy as np
from sklearn.datasets import fetch_openml


def choose(x, y, label, count, rng):
    pool = np.flatnonzero(y == label)
    if len(pool) < count:
        raise ValueError(f"Not enough examples for label {label}")
    return x[rng.choice(pool, count, replace=False)], np.full(count, label, dtype=np.int64)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=Path("data/mnist_subsets.npz"))
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)
    dataset = fetch_openml("mnist_784", version=1, as_frame=False, parser="auto")
    x = np.asarray(dataset.data, dtype=np.uint8).reshape(-1, 28, 28)
    y = np.asarray(dataset.target, dtype=np.int64)
    if len(x) != 70000:
        raise ValueError(f"Unexpected MNIST size: {len(x)}")
    train_parts = [choose(x[:60000], y[:60000], label, 50, rng) for label in (1, 7)]
    test_parts = [choose(x[60000:], y[60000:], 1, 100, rng), choose(x[60000:], y[60000:], 7, 900, rng)]
    x_train = np.concatenate([a for a,_ in train_parts]); y_train=np.concatenate([b for _,b in train_parts])
    x_test = np.concatenate([a for a,_ in test_parts]); y_test=np.concatenate([b for _,b in test_parts])
    tr=rng.permutation(len(y_train)); te=rng.permutation(len(y_test))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output,train_data=x_train[tr],train_targets=y_train[tr],test_data=x_test[te],test_targets=y_test[te])
    print(f"Wrote {args.output}: {len(tr)} training and {len(te)} test images")
    print("This is an illustrative regenerated sample, NOT the original coursework sample.")

if __name__ == "__main__":
    main()
