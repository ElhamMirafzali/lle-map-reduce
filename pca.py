from sklearn.decomposition import PCA
import load_dataset
import numpy as np


def pca_dimension_reduction(input_matrix, n_components):
    pca = PCA(n_components=n_components)
    reduced_matrix = pca.fit_transform(input_matrix)

    # Binary data
    np.save('features_reduced/features_reduced.npy', reduced_matrix)

    # Human readable data
    np.savetxt('features_reduced/features_reduced.txt', reduced_matrix)

    return reduced_matrix


# X : N by D matrix (N=60000 and D=2048) in this data set
X, labels = load_dataset.load()
pca_dimension_reduction(X, 200)
