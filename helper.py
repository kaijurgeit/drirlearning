import os
import scipy.io
import numpy as np
import tensorflow as tf


def load_data(input_dir, fileprefix='table_of_drirs_100-0',
              n_files=1, file_size=100, width=38, height=1400, cut_start=600, cut_end=2000):
    """Load data["features"] and data["labels"] from DRIR tables as mat-files from an input directory"""

    n_data = n_files * file_size
    data = {
        "features": np.zeros((n_data, width, height)),
        "labels": np.zeros((n_data, 9))
    }

    for l in range(0, n_files):
        path_data = os.path.join(input_dir, fileprefix + str(l) + '.mat')
        mat = scipy.io.loadmat(path_data)

        # Extract data and labels from mat-File
        for k in range(0, file_size):
            data["features"][l * file_size + k, :, :] = mat['table_of_drirs'][0][k][6].astype(np.float16)[:,cut_start:cut_end]
            data["labels"][l * file_size + k, 0:3] = mat['table_of_drirs'][0][k][-1][0, 0][1]  # dim
            data["labels"][l * file_size + k, 3:6] = mat['table_of_drirs'][0][k][-1][0, 0][2]  # s_pos
            data["labels"][l * file_size + k, 6:9] = mat['table_of_drirs'][0][k][-1][0, 0][3]  # r_pos

    data["features"] = data["features"].reshape((n_data, width, height, 1))

    return data


def split_data(data, split=0.8):
    """Splits data into training and test given a value from 0 to 1"""
    n_data = data["features"].shape[0]
    x_train = data["features"][:int(n_data * split), :, :, :]
    y_train = data["labels"][:int(n_data * split), :]

    x_test = data["features"][int(n_data * split):, :, :, :]
    y_test = data["labels"][int(n_data * split):, :]

    return x_train, y_train, x_test, y_test


def next_batch(batch_size, x, y):
    """Returns batches of features and labels"""
    if len(x) <= (next_batch.pointer + batch_size - 1):
        x_batch = x[next_batch.pointer:, :, :, :]
        y_batch = y[next_batch.pointer:, :]
    else:
        x_batch = x[next_batch.pointer:(next_batch.pointer + batch_size), :, :, :]
        y_batch = y[next_batch.pointer:(next_batch.pointer + batch_size), :]
    next_batch.pointer += batch_size
    return x_batch, y_batch


next_batch.pointer = 0


def conv_layer(x, filter, strides=[1, 1, 1, 1], activation=tf.nn.relu, name="conv"):
    """Wrapper for tf.nn.conv2d with summary"""
    with tf.name_scope(name):
        w = tf.Variable(tf.zeros(filter))
        b = tf.Variable(tf.zeros(filter[-1]))
        Z = tf.nn.conv2d(x, w, strides=strides)
        act = activation(Z + b)

        tf.summary.histogram("weights", w)
        tf.summary.histogram("biases", b)
        tf.summary.histogram("activations", act)
        return act


def fc_layer(x, size_out, activation=tf.nn.relu, name="fc"):
    """Wrapper for tf.matmul with summary"""
    with tf.name_scope(name):
        size_in = int(x.get_shape()[1])
        w = tf.Variable(tf.zeros(shape=[size_in, size_out]), name="w")
        b = tf.Variable(tf.zeros(shape=[size_out]), name="b")
        Z = tf.matmul(x, w)
        act = activation(Z + b)

        tf.summary.histogram("weights", w)
        tf.summary.histogram("biases", b)
        tf.summary.histogram("activations", act)
        return act
