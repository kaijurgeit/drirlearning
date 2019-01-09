"""
(1) Create your own model in model.py.
(2) Insert your model in function run() as  run_model(models.yourModel,...).
(3) Adjust configuration hard-coded or via CLI interface.
(4) Run the application.

--> type `python ./drirlearning.py -h` to get command line help.
"""

import numpy as np

import drirlearning.utils as utils
import drirlearning.models as models

np.set_printoptions(precision=3)


def run(input_dir, log_dir, n_files, n_instances, split, n_epochs, batch_size):
    """
    This function run's all models in a row. Please feel free to edit this function
    by adding/removing model's, hard-coding different hyperparameter such as the
    learning_rate, activation functions, dropouts or anything else.

    Args:
        |  input_dir (string): See run_model.
        |  log_dir (string): See load_data.
        |  n_files (int): See load_data.
        |  n_instances (int): See load_data.
        |  split (float): See run_model.
        |  n_epochs (int): See run_model.
        |  batch_size (int): See run model.

    Returns:
        |  data (dict->ndarrays): Data {'labels', 'features'}.
        |  predicted_labels (ndarray): Model's predictions after training,
        |  test_labels (ndarray): Corresponding test labels.
    """
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
    """

    Args:
        | config (dict): The project configuration, load_data and see run_model.


    Returns:
        | see run
    """
    # 1 Configuration
    c = utils.set_directories(config)
    c = utils.set_arguments(c)
    c = utils.set_config_from_cli(c)
    print(c)

    # 2 Run models
    return run(
        c['input_dir'],
        c['log_dir'],
        c['n_files'],
        c['n_instances'],
        c['split'],
        c['n_epochs'],
        c['batch_size'])


if __name__ == '__main__':
    main()
