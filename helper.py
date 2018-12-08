import os
import scipy.io
import numpy as np


def load_data(input_dir, fileprefix='table_of_drirs_100-0',
              n_files=1, file_size=100, width=38, height=1400, cut_start=600, cut_end=2000):
    n_data = n_files * file_size
    data = {
        "features": np.zeros((n_data, width, height)),
        "labels": np.zeros((n_data, 9))
    }

    for l in range(0, n_files):
        path_data = os.path.join(input_dir, fileprefix + str(l) + '.mat')
        mat = scipy.io.loadmat(path_data)
        print("i: ", l, "path_data: ", path_data)

        # Extract data and labels from mat-File
        for k in range(0, file_size):
            data["features"][l * file_size + k, :, :] = mat['table_of_drirs'][0][k][6].astype(np.float16)[:,cut_start:cut_end]
            data["labels"][l * file_size + k, 0:3] = mat['table_of_drirs'][0][k][-1][0, 0][1]  # dim
            data["labels"][l * file_size + k, 3:6] = mat['table_of_drirs'][0][k][-1][0, 0][2]  # s_pos
            data["labels"][l * file_size + k, 6:9] = mat['table_of_drirs'][0][k][-1][0, 0][3]  # r_pos

    data["features"] = data["features"].reshape((n_data, width, height, 1))

    return data


# def split_data(data, split=0.8):
#     split = 0.8
#     x_train = data["features"][:int(n_data * split), :, :, :]
#     y_train = data["labels"][:int(n_data * split), :]
#
#     x_test = data["features"][int(n_data * split):, :, :, :]
#     y_test = data["labels"][int(n_data * split):, :]
#
#     return x_train, y_train, x_test, y_test
