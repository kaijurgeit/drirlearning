import tensorflow as tf
from helpers import *


def model_cnn_2convs_4fcs(x, is_training):
    # 2 convolutional Layers
    conv1 = conv_layer(x, filter=(1, 3, 1, 4), name="conv1")
    pool1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")
    conv2 = conv_layer(pool1, filter=(7, 7, 4, 16), name="conv2")
    pool2 = tf.nn.max_pool(conv2, ksize=[1, 4, 4, 1], strides=[1, 4, 4, 1], padding="SAME")

    # 3 + 1 (output) fully connected layers
    flattened = tf.layers.Flatten()(pool2)
    fc1 = fc_layer(flattened, 6000, name="fc1")
    fc1_drop = tf.layers.dropout(fc1, rate=0.5, training=is_training)
    fc2 = fc_layer(fc1_drop, 1000, name="fc2")
    fc2_drop = tf.layers.dropout(fc2, rate=0.5, training=is_training)
    fc3 = fc_layer(fc2_drop, 200, name="fc3")
    fc3_drop = tf.layers.dropout(fc3, rate=0.5, training=is_training)
    predictions = fc_layer(fc3_drop, 9, name="predictions")

    return predictions
