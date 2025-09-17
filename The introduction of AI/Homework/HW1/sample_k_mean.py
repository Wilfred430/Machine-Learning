import numpy as np
from clustering_algos import clustering_k_mean

if __name__ == "__main__":
    # load data
    X = np.load("./features.npy")  # size: [5000, 512]

    # y = clustering(X, k=13, max_iters=20, random_state=13)
    y = clustering_k_mean(X, k=13, max_iters=300, tol=1e-6, random_state=92)  # DBSCAN 範例

    # save clustered labels
    np.save("predicted_label.npy", y)  # output size should be [5000]