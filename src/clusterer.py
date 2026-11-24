# clusterer.py
# k-means clustering logic for task 4
# run_kmeans fits the model, compute_elbow and compute_silhouette
# find the optimal k, get_cluster_centers_pixels converts centres
# back to 28x28 pixel space for visualisation

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


def run_kmeans(X, n_clusters, random_state=42):
    # fit k-means and return the fitted model and cluster labels
    km = KMeans(n_clusters=n_clusters, n_init=10, random_state=random_state)
    labels = km.fit_predict(X)
    return km, labels


def compute_elbow(X, k_range, random_state=42):
    # compute inertia for each k to produce an elbow curve
    # inertia is the total within-cluster sum of squared distances
    inertias = []
    for k in k_range:
        km, _ = run_kmeans(X, n_clusters=k, random_state=random_state)
        inertias.append(km.inertia_)
    return inertias


def compute_silhouette(X, k_range, sample_size=300, random_state=42):
    # compute mean silhouette score for each k
    # silhouette score measures how similar a sample is to its own cluster
    # compared to neighbouring clusters, ranges from -1 to 1
    # higher is better, values above 0.5 indicate decent cluster separation
    scores = []
    for k in k_range:
        _, labels = run_kmeans(X, n_clusters=k, random_state=random_state)
        try:
            score = silhouette_score(
                X, labels, sample_size=sample_size, random_state=random_state
            )
        except ValueError:
            # Fallback to computing silhouette score on the full dataset if sub-sampling fails
            score = silhouette_score(X, labels, sample_size=None)
        scores.append(round(float(score), 4))
    return scores


def get_optimal_k(k_range, silhouette_scores):
    # pick the k with the highest silhouette score
    best_idx = int(np.argmax(silhouette_scores))
    return list(k_range)[best_idx], silhouette_scores[best_idx]


def get_cluster_centers_pixels(km, scaler):
    # inverse transform cluster centers back to pixel space for visualisation
    centers = scaler.inverse_transform(km.cluster_centers_)
    centers = np.clip(centers, 0, 255)
    return centers.reshape(-1, 28, 28)


if __name__ == '__main__':
    import sys
    sys.path.insert(0, '.')
    from data_loader import load_data
    from preprocessor import flatten, get_scaler, apply_scaler

    X_train_raw, y_train, X_test_raw, y_test = load_data(
        'data/mnist_subsets.npz'
    )

    X_train_flat = flatten(X_train_raw)
    scaler       = get_scaler(X_train_flat)
    X_test_flat  = flatten(X_test_raw)
    X_test_s     = apply_scaler(scaler, X_test_flat)

    # grab only digit 7 test samples
    X_7 = X_test_s[y_test == 7]
    print(f'Digit 7 test samples: {X_7.shape[0]}')

    km100, labels100 = run_kmeans(X_7, n_clusters=100)
    print(f'K=100 inertia: {km100.inertia_:.2f}')

    k_range = range(2, 21)
    sil     = compute_silhouette(X_7, k_range)
    opt_k, opt_score = get_optimal_k(k_range, sil)
    print(f'Optimal k: {opt_k}  silhouette score: {opt_score}')
