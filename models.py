import tensorflow as tf
from helper import *


def model1(x):
    conv1 = conv_layer(x, filter=(1, 3, 1, 4), name="conv1")
    pool1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")
    conv2 = conv_layer(pool1, filter=(7, 7, 4, 16), name="conv2")
    pool2 = tf.nn.max_pool(conv2, ksize=[1, 4, 4, 1], strides=[1, 4, 4, 1], padding="SAME")

    flattened = tf.layers.Flatten()(pool2)
    fc1 = fc_layer(flattened, 6000, name="fc1")
    fc2 = fc_layer(fc1, 1000, name="fc1")
    fc3 = fc_layer(fc2, 200, name="fc1")
    predictions = fc_layer(fc3, fc3.get_shape()[1], 9, name="predictions")

    return predictions

