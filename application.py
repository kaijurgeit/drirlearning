from helper import *
import os


"""
1. Configuration
"""

np.set_printoptions(precision=3)
INPUT_DIR = "D:\\Workspace\\_Input\\"
OUTPUT_DIR = "D:\\Workspace\\_Output\\"
LOG_DIR = os.path.join(INPUT_DIR, "\\TensorBoard\\drir\\")
FILE_PREFIX = "table_of_drirs_100-0"

data = load_data(input_dir=INPUT_DIR, n_files=1, file_size=100)