import os
import argparse
import time
import scipy.io
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sound_field_analysis import gen, process


def load_data(input_dir, file_prefix='table_of_drirs_10-',
              n_files=1, n_instances=100, n_channels=38, cut_start=600, cut_end=2000):
    """
    Loads data["features"] and data["labels"] from DRIR tables as mat-files from an input directory.

    Args:
        input_dir (string): Input directory containing the .mat-Files.
        file_prefix (string): File names without the count suffix.
        n_files (int): Number of files to be loaded.
        n_instances (int): Number of instances/rows considered from the each .mat-File.
        n_channels (int): Number of microphones.
        cut_start (int): Start sample to cut each impulse response.
        cut_end (int): End sample to cut each impulse response.

    Returns:
        data (dict->(2x ndarray)): Data {'features', 'labels'}, to be used as TensorFlow input.
    """
    if cut_end <= cut_start:
        raise ValueError("cut_end must be greater than cut_start.")

    n_data = n_files * n_instances
    n_samples = cut_end - cut_start
    data = {
        "features": np.zeros((n_data, n_channels, n_samples)),
        "labels": np.zeros((n_data, 9))
    }

    for l in range(0, n_files):
        path_data = os.path.abspath(os.path.join(input_dir, file_prefix + str(l) + '.mat'))
        mat = scipy.io.loadmat(path_data)

        # Extract data and labels from mat-File
        for k in range(0, n_instances):
            data["features"][l * n_instances + k, :, :]\
                = mat['table_of_drirs'][0][k][6].astype(np.float32)[:, cut_start:cut_end]
            data["labels"][l * n_instances + k, 0:3]\
                = mat['table_of_drirs'][0][k][-1][0, 0][1]  # dim
            data["labels"][l * n_instances + k, 3:6]\
                = mat['table_of_drirs'][0][k][-1][0, 0][2]  # s_pos
            data["labels"][l * n_instances + k, 6:9]\
                = mat['table_of_drirs'][0][k][-1][0, 0][3]  # r_pos

    data["features"] = data["features"].reshape((n_data, n_channels, n_samples, 1))

    return data


def check_data_format(data):
    """Check if data format is correct"""
    if ('features' not in data.keys()) or ('labels' not in data.keys()):
        raise TypeError("data must contain 'features' and 'labels'.")
    if (type(data['features']) is not np.ndarray) or (type(data['labels']) is not np.ndarray):
        raise TypeError("Features and labels must be of type numpy.ndarray")
    if data['features'].shape[0] != data['labels'].shape[0]:
        raise TypeError("The number of instances for features and labels must be the same.")


def split_data(data, split=0.8):
    """
    Splits data into training and test.

    Args:
        data (dict->(2x ndarray)): Output of load_data or data {'features', 'labels'}.
        split (float): Split ratio from 0.0 to 1.0.

    Returns:
        x_train (ndarray): Features for training.
        y_train (ndarray): Labels for training.
        x_test (ndarray): Features for testing.
        y_test (ndarray): Labels for testing.
    """
    check_data_format(data)
    if (split < 0) or (split > 1):
        raise ValueError("The split must have a value between 0.0 and 1.0")

    n_data = data["features"].shape[0]
    x_train = data["features"][:int(n_data * split), :, :, :]
    y_train = data["labels"][:int(n_data * split), :]

    x_test = data["features"][int(n_data * split):, :, :, :]
    y_test = data["labels"][int(n_data * split):, :]

    return x_train, y_train, x_test, y_test


def shuffle(x, y):
    """
    Shuffling data is important between training epochs

    Args:
        x (ndarray): Training data features.
        y (ndarray): Training data labels.

    Returns:
        Shuffled training features (ndarray).
        Shuffled training labels (ndarray).
    """
    rnd_idx = np.random.permutation(len(x))
    return x[rnd_idx, :, :, :], y[rnd_idx, :]


def next_batch(batch_size, x, y):
    """
    Returns batches of features and labels

    Args:
        batch_size (int): Size of the batch to be processed in one epoch.
        x (ndarray): Training data features.
        y (ndarray): Training data labels.

    Returns:
        Batch of data features (ndarray).
        Batch of data labels (ndarray).
    """
    if len(x) <= (next_batch.pointer + batch_size - 1):
        x_batch = x[next_batch.pointer:, :, :, :]
        y_batch = y[next_batch.pointer:, :]
        next_batch.pointer = 0
    else:
        x_batch = x[next_batch.pointer:(next_batch.pointer + batch_size), :, :, :]
        y_batch = y[next_batch.pointer:(next_batch.pointer + batch_size), :]
    next_batch.pointer += batch_size
    assert(len(x_batch) > 0)
    return x_batch, y_batch


next_batch.pointer = 0


def conv_layer(inputs, filter, strides=[1, 1, 1, 1], activation=tf.nn.relu, name="conv"):
    """
    Wrapper for TensorFlows' tf.nn.conv2d with summary.

    Args:
        inputs (ndarray): Features.
        filter (tuple): Filter/Receptive field.
        strides (list): Strides.
        activation (callback): Activation function to use.
        name (string): Name of the TensorFlow/TensorBoard scope.

    Returns:
        Activation function (tf.Tensor)
    """
    with tf.name_scope(name):
        # TODO right stddev for initializing filter?
        # stddev = 2 / np.sqrt(np.sum(filter))
        w = tf.Variable(tf.truncated_normal(filter, stddev=0.1))
        b = tf.Variable(tf.constant(0.1, shape=[filter[-1]]))
        z = tf.nn.conv2d(inputs, w, strides=strides, padding="SAME")
        act = activation(z + b)

        tf.summary.histogram("weights", w)
        tf.summary.histogram("biases", b)
        tf.summary.histogram("activations", act)
        return act


def fc_layer(inputs, size_out, activation=tf.nn.relu, name="fc"):
    """
    Wrapper for tf.matmul with summary.

    Args:
        inputs (ndarray): Features.
        size_out (int): Number of output neurons.
        activation (callback): Activation function to use.
        name (string): Name of the TensorFlow/TensorBoard scope.

    Returns:
        Activation function (tf.Tensor)
    """
    with tf.name_scope(name):
        size_in = int(inputs.get_shape()[1])
        stddev = 2 / np.sqrt(size_in)
        w = tf.Variable(tf.truncated_normal([size_in, size_out], stddev=stddev), name="w")
        b = tf.Variable(tf.constant(0.1, shape=[size_out]))
        z = tf.matmul(inputs, w)
        act = activation(z + b)

        tf.summary.histogram("weights", w)
        tf.summary.histogram("biases", b)
        tf.summary.histogram("activations", act)
        return act


def rnn_layer(inputs, n_neurons=100, name="rnn"):
    """"""
    with tf.name_scope(name):
        cell = tf.nn.rnn_cell.LSTMCell(n_neurons)   # create a BasicRNNCell
        outputs, state = tf.nn.dynamic_rnn(cell, inputs)
        return outputs


def train(y, predictions, learning_rate):
    """

    Args:
        y (ndarray): Labels.
        predictions (ndarray): Predictions.
        learning_rate (float): Learning rate, e.g. 0.25*1E-5

    Returns:
        loss (tf.Tensor)
        minimize function
    """
    with tf.name_scope("loss"):
        loss = tf.losses.mean_squared_error(labels=y, predictions=predictions)
        # loss = tf.reduce_mean(losses)
        tf.summary.scalar("loss", loss)

    with tf.name_scope("train"):
        optimizer = tf.train.AdagradOptimizer(learning_rate=learning_rate)
        training_op = optimizer.minimize(loss)

        return loss, training_op


def leaky_relu(z, name=None):
    """Leaky ReLU might help against vanishing gradients"""
    return tf.maximum(0.01 * z, z, name=name)


def elu(z, alpha=1.0):
    """ELU might help against vanishing gradients"""
    return tf.where(z < 0, alpha * (tf.math.exp(z) - 1), z)


def selu(z,
         scale=1.0507009873554804934193349852946,
         alpha=1.6732632423543772848170429916717):
    """SELU might help against vanishing gradients"""
    return scale * elu(z, alpha)


def hparam(model, learning_rate, dropout, activation):
    """
    Gives the current run a signature by combining model's hyperparameter to a string.

    Args:
        model (callback).
        learning_rate (float).
        dropout (float).
        activation (callback).

    Returns:
        Signature (string).
    """
    act_str = activation.__name__.split('.')[-1]
    signature = "_{}-lr={}-act={}".format(model.__name__, learning_rate, act_str)
    if dropout:
        signature += "-do={}".format(dropout)
    return signature


def run_model(model, data, split, batch_size, n_epochs, learning_rate, log_dir, activation=tf.nn.relu, dropout=0):
    """
    This function is essential to the whole application.
    It runs a model given as a callback function.

    Args:
        model (callback): Each model must be created as a function in model.py.
        data (dict->(2x ndarray)): Input data of format {'features': ndarray, labels': ndarray}, see load_data
        split (float): Split ratio from 0.0 to 1.0, see split_data.
        batch_size (int): Size of the batch to be processed in one epoch, see next_batch.
        n_epochs (int): Number of epochs, each batch will be trained.
        learning_rate (float): Learning rate, e.g. 0.25*1E-5, see train.
        model_log_dir (string):
            Directory with timestamp for current run with a subdirectory for each model
            containing log data to be visualized, compared and evaluated in TensorBoard.
        activation (callback): Activation function to use.
        dropout (float): Dropout rate for convolutional nets, see tf.layers.dropout.

    Returns:
        z_test (ndarray): Predicted labels of test data.
        y_test (ndarray):  True labels of test data.
    """
    check_data_format(data)

    tf.reset_default_graph()
    is_training = tf.placeholder_with_default(True, shape=())
    model_log_dir = os.path.join(log_dir, hparam(model, learning_rate, dropout, activation))
    save_dir = model_log_dir + "\\model.ckpt"
    print(model_log_dir)

    """
    1 Graph construction phase
    """
    # 1.1 feature and label variable nodes
    x = tf.placeholder(tf.float32, shape=(None, data["features"].shape[1], data["features"].shape[2], 1), name="X")
    y = tf.placeholder(tf.float32, shape=(None, data["labels"].shape[1]), name="labels")

    # 1.2 The model itself is given as a callback function
    predictions = model(x, is_training=is_training, dropout=dropout, activation=activation)

    # 1.3 Loss, training and evaluation computation nodes
    loss, training_op = train(y, predictions, learning_rate)

    merged_summary = tf.summary.merge_all()
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

    """
    2 Execution phase
    """
    # 2.2 Prepare data
    x_train, y_train, x_test, y_test = split_data(data, split)
    n_batches = int(np.ceil(len(x_train) / batch_size))

    # 2.1 Enable GPU support
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        writer = tf.summary.FileWriter(model_log_dir, sess.graph)
        sess.run(init)

        for epoch in range(0, n_epochs):
            x_train, y_train = shuffle(x_train, y_train)
            next_batch.pointer = 0
            for batch in range(0, n_batches):
                print("batch no. ", batch)
                x_batch, y_batch = next_batch(batch_size, x_train, y_train)
                training_op_res, loss_res, summary = sess.run([training_op, loss, merged_summary],
                                                              feed_dict={x: x_batch, y: y_batch})
            if summary is not None:
                writer.add_summary(summary, epoch)

            print("epoch {}, loss {}".format(epoch, loss_res))
        try:
            z_test = predictions.eval(feed_dict={x: x_test, y: y_test, is_training: False})
        except:
            print("There must be at least one test sample, see split and file_size.")

        saver.save(sess, save_dir)

    z_test = np.reshape(z_test, (-1, 3, 3))
    y_test = np.reshape(y_test, (-1, 3, 3))

    return z_test, y_test


"""
Signal Processing
"""


def stft(data, viz=None, rate=48000, n_segs=32):
    """
    Short-time fourier transform and visualize transformed data, see scipy.signal.STFT.

    Args:
        data: See load_data.
        viz ([None, list(int)->len=4]):
            None, if no visualization desired.
            4 different instances indices (int) to be visualized.
        rate (int): Sampling rate.
        n_segs: Length of overlapping Hann-windows.

    Returns:
        data (dict->(2x ndarray)): Transformed data (instances, channels, freq bins, time bins) .
    """
    check_data_format(data)
    f, t, Zxx = scipy.signal.stft(
        data["features"][:, :, :, 0],
        fs=rate, window='hann', nperseg=n_segs, noverlap=int(n_segs / 2))

    if viz is not None:
        if len(viz) != 4:
            raise TypeError("The parameter viz must be of length 4.")
        fig1 = plt.figure()
        for i in range(0, 4):
            # STFT/Zxx: first 4 drirs and 0th node/mic -> freq/time all
            fig1.add_subplot(4, 1, i + 1).imshow(np.abs(Zxx[i, 0, :, :]))

    data["features"] = Zxx
    return data


def spat_tmp_fourier_transform(data, viz=None, rate=48000, n_segs=32, order=4):
    """
    Spatial fourier transform and visualize transformed data,
    see sound_field_analysis.process.spatFT.

    Args:
        data: See load_data.
        viz ([None, list->(4x int)):
            None, if no visualization desired.
            4 indices (int) for 3 different figures for sample visualization:
            (1) STFT - 4 different instances.
            (2) Spat. Fourier Coeffs. fixed time bins -> indices.
            (3) Spat. Fourier Coeffs. fixed freq bin.
        rate (int): Sampling rate.
        n_segs: Length of overlapping Hann-windows, see STFT.
        order (int): Spherical harmonics decomposition order, see sound_field_analysis toolbox.

    Returns:
        data (dict->(2x ndarray)): Transformed data (instances, spat base func, freq bins, time bins).
    """
    check_data_format(data)
    # 2.1 STFT
    f, t, Zxx = scipy.signal.stft(
        data["features"][:, :, :, 0],
        fs=rate, window='hann', nperseg=n_segs, noverlap=int(n_segs / 2))
    # plt.imshow(np.abs(Zxx[0, :, :]))

    n_drirs, n_nodes, n_fbins, n_tbins = Zxx.shape
    grid = gen.lebedev(order)

    # 2.2 Spatial Fourier Transform
    spat_tmp_coeffs = np.zeros((n_drirs, (order + 1) ** 2, n_fbins, n_tbins), dtype=np.complex)
    for d in range(0, n_drirs):
        for tbin in range(0, n_tbins):
            spat_tmp_coeffs[d, :, :, tbin] = process.spatFT(Zxx[d, :, :, tbin], grid, order_max=order)

    if viz is not None:
        if len(viz) != 4:
            raise TypeError("The parameter viz must be of length 4.")
        fig1 = plt.figure()
        fig2 = plt.figure()
        fig3 = plt.figure()
        for i in range(0, 4):
            # STFT/Zxx: first 4 drirs and 0th node/mic -> freq/time all
            fig1.add_subplot(4, 1, i+1).imshow(np.abs(Zxx[i, 0, :, :]))
            # spatmpFT: first 4 drirs and 4 tbins of viz -> spat/freq all
            fig2.add_subplot(4, 1, i+1).imshow(np.abs(spat_tmp_coeffs[i, :, :, viz[i]]))
            # spatmpFT: first 4 drirs and moderate fbin -> spat/tbin all
            fig3.add_subplot(4, 1, i+1).imshow(np.abs(spat_tmp_coeffs[i, :, int(n_segs/4), :]))
    data["features"] = spat_tmp_coeffs

    return data


def set_directories(config={}):
    config['file_prefix'] = config.get('file_prefix', "table_of_drirs_100-")
    config['input_dir'] = config.get('input_dir', os.path.join('.', 'data', 'input'))
    config['output_dir'] = config.get('output_dir', os.path.join('.', 'data', 'output'))
    config['log_dir'] = os.path.join(config['output_dir'],time.strftime("%Y-%m-%d_%H-%M"))
    return config


def set_arguments(config):
    config['n_files'] = 2
    config['n_instances'] = 5
    config['split'] = 0.9
    config['n_epochs'] = 5
    config['batch_size'] = 5
    return config


def set_config_from_cli(config):
    parser = argparse.ArgumentParser(
        description="Runs all Models. \n\n\
        (1) Create models in drirlearning.model.py \n\
        (2) Add models to drirlearning.py\n \
        (3) Run models from console")
    parser.add_argument('-f', help="Number of files")
    parser.add_argument('-i', help="Number of instances each file")
    parser.add_argument('-e', help="Number of epochs.")
    parser.add_argument('-b', help="Size of batch.")
    parser.add_argument('-s', help="Split.")
    args = parser.parse_args()
    if args.f is not None:
        config['n_files'] = args.e
    if args.i is not None:
        config['n_instances'] = args.i
    if args.e is not None:
        config['n_epochs'] = args.e
    if args.b is not None:
        config['batch_size'] = args.b
    if args.s is not None:
        config['split'] = args.s

    return config
