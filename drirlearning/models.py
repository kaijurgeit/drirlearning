from drirlearning.utils import *


def model_cnn_2convs_4fcs(x, is_training, dropout, activation):
    # 2 convolutional Layers
    conv1 = conv_layer(x, filter=(1, 3, 1, 4), name="conv1", activation=activation)
    pool1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")
    conv2 = conv_layer(pool1, filter=(7, 7, 4, 16), name="conv2", activation=activation)
    pool2 = tf.nn.max_pool(conv2, ksize=[1, 4, 4, 1], strides=[1, 4, 4, 1], padding="SAME")

    # 3 + 1 (output) fully connected layers
    flattened = tf.layers.Flatten()(pool2)
    fc1 = fc_layer(flattened, 6000, name="fc1", activation=activation)
    fc1_drop = tf.layers.dropout(fc1, rate=dropout, training=is_training)
    fc2 = fc_layer(fc1_drop, 1000, name="fc2", activation=activation)
    fc2_drop = tf.layers.dropout(fc2, rate=dropout, training=is_training)
    fc3 = fc_layer(fc2_drop, 200, name="fc3", activation=activation)
    fc3_drop = tf.layers.dropout(fc3, rate=dropout, training=is_training)
    predictions = fc_layer(fc3_drop, 9, name="predictions")

    return predictions


def model_cnn_3convs_4fcs(x, is_training, dropout, activation):
    # 2 convolutional Layers
    conv1 = conv_layer(x, filter=(7, 7, 1, 4), name="conv1", activation=activation)
    pool1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")
    conv2 = conv_layer(pool1, filter=(5, 5, 4, 16), name="conv2", activation=activation)
    pool2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")
    conv3 = conv_layer(pool2, filter=(5, 5, 16, 64), name="conv2", activation=activation)
    pool3 = tf.nn.max_pool(conv3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

    # 3 + 1 (output) fully connected layers
    flattened = tf.layers.Flatten()(pool3)
    fc1 = fc_layer(flattened, 4000, name="fc1", activation=activation)
    fc1_drop = tf.layers.dropout(fc1, rate=dropout, training=is_training)
    fc2 = fc_layer(fc1_drop, 1000, name="fc2", activation=activation)
    fc2_drop = tf.layers.dropout(fc2, rate=dropout, training=is_training)
    fc3 = fc_layer(fc2_drop, 200, name="fc3", activation=activation)
    fc3_drop = tf.layers.dropout(fc3, rate=dropout, training=is_training)
    predictions = fc_layer(fc3_drop, 9, name="predictions")

    return predictions


def model_cnn_2convs_5fcs(x, is_training, dropout, activation):
    # 2 convolutional Layers
    conv1 = conv_layer(x, filter=(1, 3, 1, 4), name="conv1", activation=activation)
    pool1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")
    conv2 = conv_layer(pool1, filter=(7, 7, 4, 16), name="conv2", activation=activation)
    pool2 = tf.nn.max_pool(conv2, ksize=[1, 4, 4, 1], strides=[1, 4, 4, 1], padding="SAME")

    # 3 + 1 (output) fully connected layers
    flattened = tf.layers.Flatten()(pool2)
    fc1 = fc_layer(flattened, 6000, name="fc1", activation=activation)
    fc1_drop = tf.layers.dropout(fc1, rate=dropout, training=is_training)
    fc2 = fc_layer(fc1_drop, 2000, name="fc2", activation=activation)
    fc2_drop = tf.layers.dropout(fc2, rate=dropout, training=is_training)
    fc3 = fc_layer(fc2_drop, 400, name="fc3", activation=activation)
    fc3_drop = tf.layers.dropout(fc3, rate=dropout, training=is_training)
    fc4 = fc_layer(fc3_drop, 100, name="fc3", activation=activation)
    fc4_drop = tf.layers.dropout(fc4, rate=dropout, training=is_training)
    predictions = fc_layer(fc4_drop, 9, name="predictions")

    return predictions


def model_cnn_rnn(x, is_training, dropout, activation):
    # TODO: Does not work yet
    # 2 convolutional Layers
    conv1 = conv_layer(x, filter=(3, 3, 1, 4), name="conv1", activation=activation)
    pool1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")
    conv2 = conv_layer(pool1, filter=(7, 7, 4, 16), name="conv2", activation=activation)
    pool2 = tf.nn.max_pool(conv2, ksize=[1, 4, 4, 1], strides=[1, 4, 4, 1], padding="SAME")

    flattened = tf.layers.Flatten()(pool2)
    rnn1 = rnn_layer(flattened)

    # 3 + 1 (output) fully connected layers
    fc1 = fc_layer(rnn1, 4000, name="fc1", activation=activation)
    fc1_drop = tf.layers.dropout(fc1, rate=dropout, training=is_training)
    fc2 = fc_layer(fc1_drop, 1000, name="fc2", activation=activation)
    fc2_drop = tf.layers.dropout(fc2, rate=dropout, training=is_training)
    fc3 = fc_layer(fc2_drop, 200, name="fc3", activation=activation)
    fc3_drop = tf.layers.dropout(fc3, rate=dropout, training=is_training)
    predictions = fc_layer(fc3_drop, 9, name="predictions")

    return predictions
