import numpy as np


def load():
    print("data set is loading ...")
    data_set = np.load('features/CIFAR10_incv3-keras_features.npz')
    features = data_set['features_training.npy']
    labels = data_set['labels_training.npy']
    return features, labels


