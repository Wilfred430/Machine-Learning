import numpy as np
from collections import deque

def clustering_DBSCAN(X, eps=0.5, min_samples=5):
    """
    Memory-efficient DBSCAN using only NumPy and Python built-ins.
    """
    X = np.asarray(X)
    n_samples = X.shape[0]
    eps_sq = eps * eps

    labels = np.full(n_samples, -1, dtype=int)
    visited = np.zeros(n_samples, dtype=bool)
    cluster_id = 0

    for i in range(n_samples):
        if visited[i]:
            continue
        visited[i] = True

        # 只計算 i 與所有點的距離，不建立完整矩陣
        diffs = X - X[i]
        dist_sq = np.einsum('ij,ij->i', diffs, diffs)  # 向量化平方距離
        neighbors = np.where(dist_sq <= eps_sq)[0]

        if neighbors.size < min_samples:
            continue  # 噪音
        labels[i] = cluster_id

        queue = deque(neighbors[neighbors != i])
        while queue:
            j = queue.popleft()
            if not visited[j]:
                visited[j] = True
                diffs_j = X - X[j]
                dist_sq_j = np.einsum('ij,ij->i', diffs_j, diffs_j)
                neighbors_j = np.where(dist_sq_j <= eps_sq)[0]
                if neighbors_j.size >= min_samples:
                    for k in neighbors_j:
                        if not visited[k]:
                            queue.append(k)
            if labels[j] == -1:
                labels[j] = cluster_id

        cluster_id += 1

    return labels