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
data = load_data(input_dir=INPUT_DIR, n_files=1, file_size=10)

"""
2. Run different models
"""
# Some hyperparameters
split = 0.9
n_epochs = 2
batch_size = 2

# 1.1 Learning rates
z_test, y_test = run_model(model_cnn_2convs_4fcs, data, split=split, batch_size=batch_size,
                           n_epochs=n_epochs, learning_rate=1E-4, log_dir=log_dir, dropout=True)
# 1.2 Weight initialization
# 1.3 Bias initialization
# 1.4 Activation function
# 1.5 Optimizer
# 1.6 Architecture
# 1.7 DNN type

print("Done training. Run `tensorboard --logdir={}` to see the results.".format(log_dir))