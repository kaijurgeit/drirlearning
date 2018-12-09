import os
import scipy.io
import numpy as np
import tensorflow as tf
import time


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


def conv_layer(inputs, filter, strides=[1, 1, 1, 1], activation=tf.nn.relu, name="conv"):
    """Wrapper for tf.nn.conv2d with summary"""
    with tf.name_scope(name):
        w = tf.Variable(tf.zeros(filter))
        b = tf.Variable(tf.zeros(filter[-1]))
        Z = tf.nn.conv2d(inputs, w, strides=strides)
        act = activation(Z + b)

        tf.summary.histogram("weights", w)
        tf.summary.histogram("biases", b)
        tf.summary.histogram("activations", act)
        return act


def fc_layer(inputs, size_out, activation=tf.nn.relu, name="fc"):
    """Wrapper for tf.matmul with summary"""
    with tf.name_scope(name):
        size_in = int(inputs.get_shape()[1])
        w = tf.Variable(tf.zeros(shape=[size_in, size_out]), name="w")
        b = tf.Variable(tf.zeros(shape=[size_out]), name="b")
        Z = tf.matmul(inputs, w)
        act = activation(Z + b)

        tf.summary.histogram("weights", w)
        tf.summary.histogram("biases", b)
        tf.summary.histogram("activations", act)
        return act


def train(predictions, learning_rate):
    with tf.name_scope("loss"):
        losses = tf.losses.mean_squared_error(labels=y, predictions=predictions)
        loss = tf.reduce_mean(losses)
        tf.summary.scalar("loss", loss)

    with tf.name_scope("train"):
        optimizer = tf.train.AdagradOptimizer(learning_rate=learning_rate)
        training_op = optimizer.minimize(loss)

        return loss, training_op


def eval(y, predictions):
    with tf.name_scope("eval"):
        return tf.metrics.accuracy(y, predictions)



def run_model(model, data, split, n_epochs, batch_size, learning_rate, log_dir):
    """Run a model given as a callback function"""
    tf.reset_default_graph()

    """
    1 Graph construction phase
    """
    # 1.1 feature and label variable nodes
    x = tf.placeholder(tf.float32, shape=(None, data["features"].shape[1], data["features"].shape[2], 1), name="X")
    y = tf.placeholder(tf.float32, shape=(None, data["labels"].shape[1]), name="labels")

    # 1.2 The model itself is given as a callback function
    predictions = model()

    # 1.3 Loss, training and evaluation computation nodes
    loss, training_op = train(predictions, learning_rate)
    acc, acc_op = eval(y, predictions)

    merged_summary = tf.summary.merge_all()

    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

    """
    2 Execution phase
    """
    # 2.2 Prepare data
    x_train, y_train, x_test, y_test = split_data(data, 0.9)
    batch_size = 20
    n_batches = int(np.ceil(len(x_train) / batch_size))

    # 2.1 Enable GPU support
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    with tf.Session(config=config) as sess:
        writer = tf.summary.FileWriter(log_dir + time.strftime("%Y-%m-%d_%H-%M-%S"), sess.graph)
        sess.run(init)

        for epoch in range(0, n_epochs):

            for i in range(0, n_batches // batch_size):
                x_batch, y_batch = next_batch(batch_size, x_train, y_train)
                sess.run(training_op, feed_dict={x: x_batch, y: y_batch})
                print("i: ", i)
                summary = sess.run(merged_summary, feed_dict={x: x_batch, y: y_batch})
                writer.add_summary(summary, i)

            print("epoch {}, loss {}".format(epoch, loss))
        # acc_train = sess.run(acc, feed_dict={y: y_batch})
        Z = predictions.eval(feed_dict={x: x_test, y: y_test})

    Z = np.reshape(Z, (-1, 3, 3))
    y_test = np.reshape(y_test, (-1, 3, 3))