"""
Example for loading data and saving predicted labels.
You can only use python standard library and numpy.
Do not use any other libraries.
"""
import numpy as np

def clustering(X):
    """
    Your clustering algorithm.
    """
    pass


if __name__ == "__main__":
    # load data
    X = np.load("./features.npy") # size: [5000, 512]

    y = clustering(X)

    # save clustered labels
    np.save("predicted_label.npy", y) # output size should be [5000]
