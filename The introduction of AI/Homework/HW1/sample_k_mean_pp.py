import numpy as np
import argparse
from Kpp_algos import clustering_k_mean

if __name__ == "__main__":
    # load data
    X = np.load("./features.npy")  # size: [5000, 512]

    # y = clustering(X, k=13, max_iters=20, random_state=13)
    y = clustering_k_mean(X, k=13, max_iters=300, tol=1e-6, random_state=0)  # DBSCAN 範例

    # save clustered labels
    np.save("predicted_label.npy", y)  # output size should be [5000]
    
# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--seed", type=int, default=None, help="Random seed for clustering")
#     args = parser.parse_args()

#     X = np.load("features.npy")
#     labels = clustering_k_mean(X, k=13, random_state=args.seed)  # k-means++ 範例
#     np.save("predicted_label.npy", labels)
