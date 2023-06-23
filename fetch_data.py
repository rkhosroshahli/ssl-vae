import numpy as np
from tensorflow import keras


def fetch_mnist_digit(ndim):
    # Load the data and split it between train and test sets
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    # Scale images to the [0, 1] range
    # x_train = x_train.astype("float32") / 255
    # x_test = x_test.astype("float32") / 255
    # Make sure images have shape (28, 28, 1)
    x_train = x_train.reshape(x_train.shape[0], ndim, ndim, 1).astype("float32") / 255.0
    x_test = x_test.reshape(x_test.shape[0], ndim, ndim, 1).astype("float32") / 255.0
    print("x_train shape:", x_train.shape)
    print(x_train.shape[0], "train samples")
    print(x_test.shape[0], "test samples")

    return (x_train, y_train), (x_test, y_test)
