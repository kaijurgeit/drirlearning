"""
Try out different models and hyperparameters
--------------------------------------------
# 1 Learning rates
# 2 Weight initialization
# 3 Bias initialization
# 4 Activation function
# 5 Optimizer
# 6 Architecture
# 7 DNN type
"""

import numpy as np

import drirlearning.utils as utils
import drirlearning.models as models

np.set_printoptions(precision=3)


def run(input_dir, log_dir, n_files, n_instances, split, n_epochs, batch_size):
    """Specify Models to run"""
    # 1 Load the data
    data = utils.load_data(input_dir=input_dir, n_files=n_files, n_instances=n_instances)
    i = 0
    predicted_labels = [0] * 100
    test_labels = [0] * 100

    # Model 1: Dropout
    predicted_labels[i], test_labels[i] = utils.run_model(
        models.cnn_2convs_4fcs, data, split=split, batch_size=batch_size,
        n_epochs=n_epochs, learning_rate=0.25*1E-5, log_dir=log_dir, dropout=0.2)
    i += 1

    # Model 1: Activation
    predicted_labels[i], test_labels[i] = utils.run_model(
        models.cnn_2convs_4fcs, data, split=split, batch_size=batch_size,
        n_epochs=n_epochs, learning_rate=0.25*1E-5, log_dir=log_dir, activation=utils.elu, dropout=0.2)
    i += 1

    # Model 2: Learning rate:
    predicted_labels[i], test_labels[i] = utils.run_model(
        models.cnn_2convs_5fcs, data, split=split, batch_size=batch_size,
        n_epochs=n_epochs, learning_rate=0.25*1E-5, log_dir=log_dir, dropout=0.2)
    i += 1

    predicted_labels[i], test_labels[i] = utils.run_model(
        models.cnn_2convs_5fcs, data, split=split, batch_size=batch_size,
        n_epochs=n_epochs, learning_rate=0.25*1E-5, log_dir=log_dir, activation=utils.elu, dropout=0.2)
    i += 1

    # Model 3
    predicted_labels[i], test_labels[i] = utils.run_model(
        models.cnn_3convs_4fcs, data, split=split, batch_size=batch_size,
        n_epochs=n_epochs, learning_rate=0.5*1E-5, log_dir=log_dir, activation=utils.elu, dropout=0.2)
    i += 1

    print("Done training. Run `tensorboard --logdir={}` to see the results.".format(log_dir))

    data = utils.spat_tmp_fourier_transform(data, viz=[20, 40, 60, 80])
    # data = utils.stft(data, viz=True)
    return data, predicted_labels, test_labels


def main(config={}):
    # 1 Configuration
    c = utils.set_directories(config)
    c = utils.set_hyperparameters(c)
    c = utils.set_config_from_cli(c)
    print(c)

    # 2 Run models
    run(c)


    # z = [0] * 20
    # y_test = [0] * 20




    pass




if __name__ == '__main__':
    main()
