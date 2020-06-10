import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np


def show_data(input_matrix):
    if len(input_matrix) == 2:
        n = len(input_matrix[0])
        data_set = np.load('features/CIFAR10_incv3-keras_features.npz')
        labels = data_set['labels_training.npy'][:n]

        plt.xlabel('x')
        plt.ylabel('y')
        colors = np.random.rand(10)
        colors = [colors[labels[i]] for i in range(n)]

        plt.scatter(input_matrix[0], input_matrix[1], c=colors, edgecolors='k')
        plt.grid(axis='x', linestyle='-', color='#777777')
        plt.grid(axis='y', linestyle='-', color='#777777')
        plt.show()

    elif len(input_matrix) == 3:
        n = len(input_matrix[0])
        data_set = np.load('features/CIFAR10_incv3-keras_features.npz')
        labels = data_set['labels_training.npy'][:n]

        # show 2d
        plt.xlabel('x')
        plt.ylabel('y')
        colors = np.random.rand(10)
        colors = [colors[labels[i]] for i in range(n)]

        plt.scatter(input_matrix[0], input_matrix[1], c=colors, edgecolor='k')
        plt.grid(axis='x', linestyle='-', color='#777777')
        plt.grid(axis='y', linestyle='-', color='#777777')
        plt.show()

        # show 3d
        fig = plt.figure(1, figsize=(10, 10))
        ax = Axes3D(fig)
        colors = np.random.rand(10)
        colors = [colors[labels[i]] for i in range(n)]
        ax.scatter(input_matrix[0, :], input_matrix[1, :], input_matrix[2, :],
                   edgecolor='k', s=50, c=colors)
        ax.set_title("3D visualization")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")

        plt.show()

    else:
        print("sorry dimension is more than 3! \n There is no implementation for it")
