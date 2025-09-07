import numpy as np

def clustering(X, K=10, max_iters=100, tol=1e-4):
    """
    K-Means clustering algorithm.
    Parameters:
        X : ndarray of shape (N, D) - input features
        K : int - number of clusters
        max_iters : int - maximum number of iterations
        tol : float - convergence threshold
    Returns:
        labels : ndarray of shape (N,) - predicted cluster labels
    """
    N, D = X.shape

    # Step 1: Randomly initialize K centroids
    rng = np.random.default_rng(seed=42)
    centroids = X[rng.choice(N, K, replace=False)]

    for _ in range(max_iters):
        # Step 2: Assign each point to the nearest centroid
        distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)  # shape: (N, K)
        labels = np.argmin(distances, axis=1)  # shape: (N,)

        # Step 3: Update centroids
        new_centroids = np.array([X[labels == k].mean(axis=0) if np.any(labels == k) else centroids[k] for k in range(K)])

        # Step 4: Check for convergence
        shift = np.linalg.norm(new_centroids - centroids)
        if shift < tol:
            break

        centroids = new_centroids

    return labels

if __name__ == "__main__":
    # load data
    X = np.load("./features.npy")  # shape: [5000, 512]

    # run clustering
    y = clustering(X, K=10)

    # save clustered labels
    np.save("predicted_label.npy", y)  # shape: [5000]