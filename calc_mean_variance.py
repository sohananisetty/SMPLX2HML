from os.path import join as pjoin

import numpy as np
import os
from tqdm import tqdm


def findAllFile(base):
    """
    Recursively find all files in the specified directory.

    Args:
        base (str): The base directory to start the search.

    Returns:
        list: A list of file paths found in the directory and its subdirectories.
    """
    file_path = []
    for root, ds, fs in os.walk(base, followlinks=True):
        for f in fs:
            fullname = os.path.join(root, f)
            file_path.append(fullname)
    return file_path


# root_rot_velocity (B, seq_len, 1)
# root_linear_velocity (B, seq_len, 2)
# root_y (B, seq_len, 1)
# ric_data (B, seq_len, (joint_num - 1)*3)
# rot_data (B, seq_len, (joint_num - 1)*6)
# local_velocity (B, seq_len, joint_num*3)
# foot contact (B, seq_len, 4)
# def mean_variance(data_dir, save_dir, joints_num):


if __name__ == "__main__":
    joints_num = 52

    # change your motion_data
    data_dir = "/srv/hays-lab/scratch/sanisetty3/motionx/motion_data/new_joint_vecs/"
    save_dir = "/srv/hays-lab/scratch/sanisetty3/motionx/motion_data/"

    file_list = findAllFile(data_dir)

    data_list = []

    for file in tqdm(file_list):
        data = np.load(file)
        # data[:,292:310] = np.nan_to_num(data[:,292:310] , nan = 0)
        # print(data.shape)
        if np.isnan(data).any():
            print(file)
            continue
        data_list.append(data)

    data = np.concatenate(data_list, axis=0)
    # print(data.shape)
    Mean = data.mean(axis=0)
    Std = data.std(axis=0)
    Std[0:1] = Std[0:1].mean() / 1.0
    Std[1:3] = Std[1:3].mean() / 1.0
    Std[3:4] = Std[3:4].mean() / 1.0
    Std[4 : 4 + (joints_num - 1) * 3] = Std[4 : 4 + (joints_num - 1) * 3].mean() / 1.0
    Std[4 + (joints_num - 1) * 3 : 4 + (joints_num - 1) * 9] = (
        Std[4 + (joints_num - 1) * 3 : 4 + (joints_num - 1) * 9].mean() / 1.0
    )
    Std[4 + (joints_num - 1) * 9 : 4 + (joints_num - 1) * 9 + joints_num * 3] = (
        Std[4 + (joints_num - 1) * 9 : 4 + (joints_num - 1) * 9 + joints_num * 3].mean()
        / 1.0
    )
    Std[4 + (joints_num - 1) * 9 + joints_num * 3 :] = (
        Std[4 + (joints_num - 1) * 9 + joints_num * 3 :].mean() / 1.0
    )

    assert 8 + (joints_num - 1) * 9 + joints_num * 3 == Std.shape[-1]
    print(Mean.shape, Std.shape)

    np.save(pjoin(save_dir, "Mean.npy"), Mean)
    np.save(pjoin(save_dir, "Std.npy"), Std)

    # return Mean, Std
