import numpy as np
from numpy import linalg as LA
from sklearn.neighbors import KDTree


# This method is not used.
def k_nearest_neighbour(x, k):
    print("computing k nearest neighbours ...")
    tree = KDTree(x)
    neighbours = np.array([[0] * (k + 1)])

    for i in range(len(x)):
        dist, ind = tree.query(x[i:i + 1], k + 1)
        neighbours = np.append(neighbours, ind, axis=0)

    neighbours = neighbours[1:]

    f = open('save.txt', 'w')
    a = np.array2string(neighbours, formatter={'float_kind': lambda n: "%.2f" % n})
    f.write(a)
    f.close()

    return neighbours


# solve for reconstruction weights W
def reconstruct_weight(x, neighbours, k):
    x = np.transpose(x)
    weight = np.zeros([len(x), len(x)])
    error = 0
    for i in range(len(x)):

        # create matrix z
        z = np.array([[0] * len(x[0])])
        x_ith_neighbours = neighbours[i:i + 1]

        for j in x_ith_neighbours[0, :]:
            z = np.append(z, x[j:j + 1], axis=0)
        z = z[1:]

        # subtract Xi from every column of z
        z = np.subtract(x[i: i + 1], z)

        z = np.transpose(z)

        # compute the local covariance
        c = np.matmul(np.transpose(z), z)

        # solve linear system
        identity = np.ones((len(c), 1))
        w = np.linalg.solve(c, identity)

        sum_w = np.sum(w)
        sum_multiply_w_x = np.array([[0] * len(x[0])])
        for j in x_ith_neighbours[0, :]:
            w_index = list(x_ith_neighbours[0, :]).index(j)
            weight[i, j:j + 1] = w[w_index][0] / sum_w

        # calculate error
        for j in range(len(x)):
            zarb = weight[i, j:j + 1] * x[j:j + 1]
            np.add(sum_multiply_w_x, zarb[0])

        error += (np.linalg.norm(np.subtract(x[i: i + 1], sum_multiply_w_x), 'fro'))
    return weight, error


def embedding_coordinate(weight, dimension):
    n = len(weight)
    identity = np.identity(n)
    i_sub_w = np.subtract(identity, weight)
    M = np.multiply(np.transpose(i_sub_w), i_sub_w)
    eigenvalues, eigenvectors = LA.eigh(M)
    smallest_eigenvalues_index = eigenvalues.argsort()[:dimension + 1]
    Y = np.array([[]])
    for q in smallest_eigenvalues_index[1:]:
        Y = np.append(Y, eigenvectors[q])

    Y = np.reshape(Y, (dimension, n))
    return Y
