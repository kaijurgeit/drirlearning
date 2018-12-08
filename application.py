from helper import *
import numpy as np
import os


"""
1. Configuration
"""

np.set_printoptions(precision=3)
INPUT_DIR = "D:\\Workspace\\_Input\\"
OUTPUT_DIR = "D:\\Workspace\\_Output\\"
LOG_DIR = os.path.join(INPUT_DIR, "\\TensorBoard\\drir\\")
FILE_PREFIX = "table_of_drirs_100-0"

data = load_data(input_dir=INPUT_DIR, n_files=1, file_size=51)
x_train, y_train, x_test, y_test = split_data(data, 0.9)

batch_size = 20
n_batches = int(np.ceil(len(x_train) / batch_size))

for batch in range(0, n_batches):
    x_batch, y_batch = next_batch(batch_size, x_train, y_train)
    print(batch, ": ", x_batch.shape)