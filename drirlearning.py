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
import os
import time
import argparse
import numpy as np

# import drirlearning.utils as utils
# import drirlearning.models as models

np.set_printoptions(precision=3)


def set_directories(config={}):
    config['input_dir'] = os.path.join('.', 'data', 'input')
    config['output_dir'] = os.path.join('.', 'data', 'output')
    config['log_dir'] = config['output_dir'] + time.strftime("%Y-%m-%d_%H-%M") + "\\"
    config['file_prefix'] = "table_of_drirs_100-"
    return config


def set_hyperparameters(config):
    config['split'] = 0.9
    config['n_epochs'] = 20
    config['batch_size'] = 25
    return config


def set_config_from_cli(config):
    parser = argparse.ArgumentParser(
        description="Runs all Models. \n\n\
        (1) Create models in drirlearning.model.py \n\
        (2) Add models to drirlearning.py\n \
        (3) Run models from console")
    parser.add_argument('-s', help="Split.")
    parser.add_argument('-e', help="Number of epochs.")
    parser.add_argument('-b', help="Size of batch.")
    args = parser.parse_args()
    if args.s is not None:
        config['split'] = args.s
    if args.e is not None:
        config['n_epochs'] = args.e
    if args.b is not None:
        config['batch_size'] = args.b

    return config, parser



def main():
    # 1 Configuration
    c = set_directories()
    c = set_hyperparameters(c)
    c, parser = set_config_from_cli(c)

    z = [0] * 20
    y_test = [0] * 20

    # 3 Parse arguments from CLI
    print(c)


    # print("split={}, n_epochs={}, batch_size={}".format(split, n_epochs, batch_size))

    pass


# def main2():
#     """Specify Models to run"""
#     # 1 Load the data
#     data = utils.load_data(input_dir=INPUT_DIR, n_files=1, n_instances=10)
#
#
#     i = 0
#
#     # Model 4:
#     data = utils.spat_tmp_fourier_transform(data, viz=[20, 40, 60, 80])
#     data = utils.stft(data, viz=True)
#
#     # Model 1: Dropout
#     z[i], y_test[i] = utils.run_model(
#         models.model_cnn_2convs_4fcs, data, split=split, batch_size=batch_size,
#         n_epochs=n_epochs, learning_rate=0.25*1E-5, log_dir=log_dir, dropout=0.2)
#     i += 1
#
#     # Model 1: Activation
#     z[i], y_test[i] = utils.run_model(
#         models.model_cnn_2convs_4fcs, data, split=split, batch_size=batch_size,
#         n_epochs=n_epochs, learning_rate=0.25*1E-5, log_dir=log_dir, activation=elu, dropout=0.2)
#     i += 1
#
#     # Model 2: Learning rate:
#     z[i], y_test[i] = utils.run_model(
#         models.model_cnn_2convs_5fcs, data, split=split, batch_size=batch_size,
#         n_epochs=n_epochs, learning_rate=0.25*1E-5, log_dir=log_dir, dropout=0.2)
#     i += 1
#
#     z[i], y_test[i] = utils.run_model(
#         models. model_cnn_2convs_5fcs, data, split=split, batch_size=batch_size,
#         n_epochs=n_epochs, learning_rate=0.25*1E-5, log_dir=log_dir, activation=elu, dropout=0.2)
#     i += 1
#
#     # Model 3
#     z[i], y_test[i] = utils.run_model(
#         models.model_cnn_3convs_4fcs, data, split=split, batch_size=batch_size,
#         n_epochs=n_epochs, learning_rate=0.5*1E-5, log_dir=log_dir, activation=elu, dropout=0.2)
#     i += 1
#
#     print("Done training. Run `tensorboard --logdir={}` to see the results.".format(log_dir))
#     pass


if __name__ == '__main__':
    main()
