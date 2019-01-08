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
import drirlearning.utils as utils
import drirlearning.models as models
import numpy as np
import time


"""
1. Configuration
"""
np.set_printoptions(precision=3)
INPUT_DIR = "D:\\Workspace\\_Input\\"
OUTPUT_DIR = "D:\\Workspace\\_Output\\"
LOG_ROOT = OUTPUT_DIR + "TensorBoard\\drir\\"
FILE_PREFIX = "table_of_drirs_100-0"

log_dir = LOG_ROOT + time.strftime("%Y-%m-%d_%H-%M") + "\\"

# Some hyperparameters
split = 0.9
n_epochs = 20
batch_size = 25

z = [0] * 20
y_test = [0] * 20


def main():
    """Specify Models to run"""
    # 1 Load the data
    data = utils.load_data(input_dir=INPUT_DIR, n_files=1, n_instances=10)


    i = 0

    # Model 4:
    data = utils.spat_tmp_fourier_transform(data, viz=[20, 40, 60, 80])
    data = utils.stft(data, viz=True)

    # Model 1: Dropout
    z[i], y_test[i] = utils.run_model(
        models.model_cnn_2convs_4fcs, data, split=split, batch_size=batch_size,
        n_epochs=n_epochs, learning_rate=0.25*1E-5, log_dir=log_dir, dropout=0.2)
    i += 1

    # Model 1: Activation
    z[i], y_test[i] = utils.run_model(
        models.model_cnn_2convs_4fcs, data, split=split, batch_size=batch_size,
        n_epochs=n_epochs, learning_rate=0.25*1E-5, log_dir=log_dir, activation=elu, dropout=0.2)
    i += 1

    # Model 2: Learning rate:
    z[i], y_test[i] = utils.run_model(
        models.model_cnn_2convs_5fcs, data, split=split, batch_size=batch_size,
        n_epochs=n_epochs, learning_rate=0.25*1E-5, log_dir=log_dir, dropout=0.2)
    i += 1

    z[i], y_test[i] = utils.run_model(
        models. model_cnn_2convs_5fcs, data, split=split, batch_size=batch_size,
        n_epochs=n_epochs, learning_rate=0.25*1E-5, log_dir=log_dir, activation=elu, dropout=0.2)
    i += 1

    # Model 3
    z[i], y_test[i] = utils.run_model(
        models.model_cnn_3convs_4fcs, data, split=split, batch_size=batch_size,
        n_epochs=n_epochs, learning_rate=0.5*1E-5, log_dir=log_dir, activation=elu, dropout=0.2)
    i += 1

    print("Done training. Run `tensorboard --logdir={}` to see the results.".format(log_dir))
    pass


if __name__ == '__main__':
    main()
