import numpy as np

def clustering_k_mean(X, k=13, max_iters=300, tol=1e-6, random_state=92):
    """
    K-means++ clustering using only NumPy.
    Returns only the cluster labels.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
        Input data.
    k : int
        Number of clusters.
    max_iters : int
        Maximum number of iterations.
    tol : float
        Tolerance for convergence (based on center movement).
    random_state : int or None
        Random seed for reproducibility.

    Returns
    -------
    labels : ndarray of shape (n_samples,)
        Cluster label for each point.
    """
    if random_state is not None:
        np.random.seed(random_state)

    n_samples, n_features = X.shape

    # --- Step 1: k-means++ initialization ---
    centers = np.empty((k, n_features), dtype=X.dtype)

    # 1.1 隨機選第一個中心
    first_idx = np.random.randint(0, n_samples)
    centers[0] = X[first_idx]

    # 1.2 根據距離平方的機率分佈選取剩下的中心
    for i in range(1, k):
        dist_sq = np.min(
            np.square(X[:, np.newaxis] - centers[np.newaxis, :i]).sum(axis=2),
            axis=1
        )
        probs = dist_sq / dist_sq.sum()
        next_idx = np.random.choice(n_samples, p=probs)
        centers[i] = X[next_idx]

    # --- Step 2: Lloyd's algorithm ---
    for _ in range(max_iters):
        # 2.1 分配每個點到最近的中心
        distances = np.linalg.norm(X[:, np.newaxis] - centers[np.newaxis, :], axis=2)
        labels = np.argmin(distances, axis=1)

        # 2.2 計算新的中心
        new_centers = np.array([
            X[labels == j].mean(axis=0) if np.any(labels == j) else centers[j]
            for j in range(k)
        ])

        # 2.3 檢查收斂
        shift = np.linalg.norm(new_centers - centers, axis=1).max()
        centers = new_centers
        if shift < tol:
            break

    return labels