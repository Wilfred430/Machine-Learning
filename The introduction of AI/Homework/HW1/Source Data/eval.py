"""
Please do not include this file in your submission.
This file is only for your reference to evaluate your clustering results.
"""

import numpy as np
from sklearn.metrics import silhouette_score, adjusted_rand_score
import sys

if __name__ == "__main__":
    """Usage: python eval.py {your predicted file path}"""
    file_name = sys.argv[1]

    cluster_label = np.load(file_name)
    true_label = np.load("label_test.npy")
    data = np.load("features.npy")

    silhouette_avg = silhouette_score(data, cluster_label)
    print(f'Silhouette Coefficient: {silhouette_avg:.3f}')

    # Final ARI score will calculated using whole 500 data.
    test_num = true_label.size
    ari_score = adjusted_rand_score(true_label, cluster_label[:test_num])
    print(f'Adjusted Rand Index (ARI): {ari_score:.3f}')
