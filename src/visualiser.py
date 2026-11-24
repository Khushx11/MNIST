# visualiser.py
# plotting functions for task 1 visualisations
# sample image grid, pixel stats plots, class distribution bar chart
# all functions accept an optional save_path to write the figure to disk

import numpy as np
import matplotlib.pyplot as plt


def plot_sample_images(X_raw, y, n_per_class=8, save_path=None):
    # show n_per_class examples from each digit side by side
    fig, axes = plt.subplots(2, n_per_class, figsize=(n_per_class * 1.6, 4))
    fig.suptitle(
        'Sample training images  |  top row: digit 1,  bottom row: digit 7',
        fontsize=11
    )
    for row, label in enumerate([1, 7]):
        idx = np.where(y == label)[0][:n_per_class]
        for col, i in enumerate(idx):
            ax = axes[row, col]
            ax.imshow(X_raw[i], cmap='gray_r', interpolation='nearest')
            ax.axis('off')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=120)
    plt.show()


def plot_pixel_stats(X_raw, y, save_path=None):
    # mean image, std image, and non-zero pixel histogram per class
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    for row, (label, name) in enumerate(zip([1, 7], ['Digit 1', 'Digit 7'])):
        imgs = X_raw[y == label].astype(float)

        axes[row, 0].imshow(imgs.mean(axis=0), cmap='hot', interpolation='nearest')
        axes[row, 0].set_title(f'Mean image  {name}', fontsize=10)
        axes[row, 0].axis('off')

        axes[row, 1].imshow(imgs.std(axis=0), cmap='viridis', interpolation='nearest')
        axes[row, 1].set_title(f'Pixel std  {name}', fontsize=10)
        axes[row, 1].axis('off')

        px = imgs.flatten()
        px_nonzero = px[px > 0]
        axes[row, 2].hist(px_nonzero, bins=40, color='steelblue',
                          edgecolor='white', alpha=0.85)
        axes[row, 2].set_title(f'Non-zero pixel intensities  {name}', fontsize=10)
        axes[row, 2].set_xlabel('Pixel value (0 to 255)')
        axes[row, 2].set_ylabel('Count')

    plt.suptitle('Per-class pixel statistics (training set)', fontsize=12)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=120)
    plt.show()


def plot_class_distribution(y_train, y_test, save_path=None):
    # bar chart comparing class counts in train vs test split
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    for ax, (y, title) in zip(axes, [
            (y_train, f'Training set  (n={len(y_train)})'),
            (y_test,  f'Test set  (n={len(y_test)})')]):
        counts = {l: int((y == l).sum()) for l in [1, 7]}
        bars = ax.bar(
            ['Digit 1', 'Digit 7'],
            list(counts.values()),
            color=['#4C72B0', '#DD8452'],
            edgecolor='white',
            width=0.5
        )
        ax.set_title(title, fontsize=11)
        ax.set_ylabel('Number of samples')
        for bar, val in zip(bars, counts.values()):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.4,
                str(val), ha='center', va='bottom', fontweight='bold'
            )
    plt.suptitle('Class distribution across splits', fontsize=12)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=120)
    plt.show()


def plot_lr_weights(model, save_path=None):
    # reshape the logistic regression weight vector back to 28x28
    # positive values (warm colours) push the prediction toward digit 7
    # negative values (cool colours) push toward digit 1
    weights = model.coef_[0].reshape(28, 28)
    fig, ax = plt.subplots(figsize=(5, 5))
    im = ax.imshow(weights, cmap='RdBu_r', interpolation='nearest')
    ax.set_title(
        'Logistic regression weights reshaped to 28x28\n'
        'red = pushes toward digit 7  |  blue = pushes toward digit 1',
        fontsize=10
    )
    ax.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=120)
    plt.show()
