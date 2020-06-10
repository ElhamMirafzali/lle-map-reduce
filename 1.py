from keras.applications.inception_v3 import InceptionV3

model = InceptionV3(include_top=False, weights='imagenet', \
                    input_tensor=None, input_shape=None, pooling=None, classes=1000)

# ____________________________________________________Selected network:

network_names = ['incv3', 'resnet50', 'vgg16', 'vgg19']

print("Available networks = ", network_names)
cnnid = int(input("Please choose the CNN network [0-{n}]: ".format(n=len(network_names) - 1)))
selected_network = network_names[cnnid]
print("Selected network: ", selected_network)


import time
import myutils
import numpy as np
import tensorflow as tf
from keras.layers import Input, Dense, AveragePooling2D, GlobalAveragePooling2D
from keras import backend as K

from keras.datasets import cifar10

(X_train, y_train), (X_test, y_test) = cifar10.load_data()

X_train = np.concatenate((X_train, X_test))
y_train = np.concatenate((y_train, y_test))

n_training = X_train.shape[0]

y_train = y_train.flatten()

print(X_train.shape, y_train.shape)

from matplotlib import pyplot as plt

plt.imshow(X_train[0])
plt.show()

from keras.models import Model
from keras.applications.inception_v3 import InceptionV3
from keras.applications.resnet50 import ResNet50
from keras.applications.vgg16 import VGG16
from keras.applications.vgg19 import VGG19

input_shape = {
    'incv3': (299, 299, 3),
    'resnet50': (224, 224, 3),
    'vgg16': (224, 224, 3),
    'vgg19': (224, 224, 3)
}[selected_network]


def create_model_incv3():
    tf_input = Input(shape=input_shape)
    model = InceptionV3(input_tensor=tf_input, weights='imagenet', include_top=False)
    output_pooled = AveragePooling2D((8, 8), strides=(8, 8))(model.output)
    return Model(model.input, output_pooled)


def create_model_resnet50():
    tf_input = Input(shape=input_shape)
    return ResNet50(input_tensor=tf_input, include_top=False)


def create_model_vgg16():
    tf_input = Input(shape=input_shape)
    model = VGG16(input_tensor=tf_input, include_top=False)
    output_pooled = AveragePooling2D((7, 7))(model.output)
    return Model(model.input, output_pooled)


def create_model_vgg19():
    tf_input = Input(shape=input_shape)
    model = VGG19(input_tensor=tf_input, include_top=False)
    output_pooled = AveragePooling2D((7, 7))(model.output)
    return Model(model.input, output_pooled)


create_model = {
    'incv3': create_model_incv3,
    'resnet50': create_model_resnet50,
    'vgg16': create_model_vgg16,
    'vgg19': create_model_vgg19
}[selected_network]

# tensorflow placeholder for batch of images from CIFAR10 dataset
batch_of_images_placeholder = tf.placeholder("uint8", (None, 32, 32, 3))

batch_size = {
    'incv3': 16,
    'resnet50': 16,
    'vgg16': 16,
    'vgg19': 16
}[selected_network]

# Inception default size is 299x299
tf_resize_op = tf.image.resize_images(batch_of_images_placeholder, (input_shape[:2]), method=0)

# data generator for tensorflow session
from keras.applications.inception_v3 import preprocess_input as incv3_preprocess_input
from keras.applications.resnet50 import preprocess_input as resnet50_preprocess_input
from keras.applications.vgg16 import preprocess_input as vgg16_preprocess_input
from keras.applications.vgg19 import preprocess_input as vgg19_preprocess_input

preprocess_input = {
    'incv3': incv3_preprocess_input,
    'resnet50': resnet50_preprocess_input,
    'vgg16': vgg16_preprocess_input,
    'vgg19': vgg19_preprocess_input
}[selected_network]


def data_generator(sess, data, labels):
    def generator():
        start = 0
        end = start + batch_size
        n = data.shape[0]
        while True:
            batch_of_images_resized = sess.run(tf_resize_op, {batch_of_images_placeholder: data[start:end]})
            batch_of_images__preprocessed = preprocess_input(batch_of_images_resized)
            batch_of_labels = labels[start:end]
            start += batch_size
            end += batch_size
            if start >= n:
                start = 0
                end = batch_size
            yield (batch_of_images__preprocessed, batch_of_labels)

    return generator


with tf.Session() as sess:
    # setting tensorflow session to Keras
    K.set_session(sess)
    # setting phase to training
    K.set_learning_phase(0)  # 0 - test,  1 - train

    model = create_model()

    data_train_gen = data_generator(sess, X_train, y_train)
    ftrs_training = model.predict_generator(data_train_gen(), n_training / batch_size, verbose=1)

features_training = np.array([ftrs_training[i].flatten() for i in range(n_training)])

np.savez_compressed("features/CIFAR10_{}-keras_features.npz".format(selected_network), \
                    features_training=features_training, \
                    labels_training=y_train)

features_training.shape
print('Ten first features of X_train[0] (see figure above, with the frog)')
features_training[0][0:10]
