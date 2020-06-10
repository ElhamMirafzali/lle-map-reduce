import map_reduce
import lle
import visualization
import numpy as np

if __name__ == '__main__':
    matrix = np.load("features_reduced/features_reduced.npy")
    matrix = matrix[:1000]
    matrix = np.transpose(matrix)

    # matrix = np.array([[1, 2, 3, 4, 5], [1, 3, 2, 4, 5], [6, 7, 8, 9, 10], [6, 6, 8, 9, 10]])
    # matrix = np.transpose(matrix)

    # matrix is D X N
    k = 20
    d = 3
    map_num = 4
    neighbours, exe_time = map_reduce.find_k_nearest_neighbours(matrix, k, map_num)

    # neighbours, exe_time = map_reduce.new_map_reduce(matrix, k, map_num)
    print("exe_time = \n", exe_time)

    # matrix = np.transpose(matrix)
    weight_matrix, error = lle.reconstruct_weight(matrix, neighbours, k)
    print("Error = ", error)
    result = lle.embedding_coordinate(weight_matrix, d)
    visualization.show_data(result)
