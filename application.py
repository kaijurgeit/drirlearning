from helpers import *
from models import *
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

run_model(model1, data, split=0.9, n_epochs=5, batch_size=20, learning_rate=1E-2,
          log_dir=LOG_DIR)
