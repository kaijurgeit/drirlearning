from helpers import *
from models import *
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
data = load_data(input_dir=INPUT_DIR, n_files=10, file_size=100)

"""
2. Run different models
"""
# Some hyperparameters
split = 0.9
n_epochs = 40
batch_size = 25

z = [0] * 20
y_test = [0] * 20

i = 0


# Model 1: Dropout
z[i], y_test[i] = run_model(
    model_cnn_2convs_4fcs, data, split=split, batch_size=batch_size,
    n_epochs=n_epochs, learning_rate=0.25*1E-5, log_dir=log_dir, dropout=0.2)
i += 1

# Model 1: Activation
z[i], y_test[i] = run_model(
    model_cnn_2convs_4fcs, data, split=split, batch_size=batch_size,
    n_epochs=n_epochs, learning_rate=0.25*1E-5, log_dir=log_dir, activation=elu, dropout=0.2)
i += 1


# Model 2: Learning rate:
z[i], y_test[i] = run_model(
    model_cnn_2convs_5fcs, data, split=split, batch_size=batch_size,
    n_epochs=n_epochs, learning_rate=0.25*1E-5, log_dir=log_dir, dropout=0.2)
i += 1

z[i], y_test[i] = run_model(
    model_cnn_2convs_5fcs, data, split=split, batch_size=batch_size,
    n_epochs=n_epochs, learning_rate=0.25*1E-5, log_dir=log_dir, activation=elu, dropout=0.2)
i += 1


# 1.1 Learning rates
# 1.2 Weight initialization
# 1.3 Bias initialization
# 1.4 Activation function
# 1.5 Optimizer
# 1.6 Architecture
# 1.7 DNN type

print("Done training. Run `tensorboard --logdir={}` to see the results.".format(log_dir))