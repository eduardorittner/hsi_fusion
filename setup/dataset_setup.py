# After having downloaded the .hdf train and val files, split each one into 3 different files:
# input:
# rgb [1024, 1024, 4]
# msi [256, 256, 61]
# output:
# msi [1024, 1024, 61]

import numpy as np
import h5py
import argparse
from os import path
from tqdm import tqdm

def save2np(file, dest_path):
    with h5py.File(file) as f:
        keys = list(f.keys())

        for key in tqdm(keys):
            arr = f[key][()]
            rgb_in = arr[:4].T
            msi_out_vis = arr[4 : 4 + 31]
            msi_out_nir = arr[31 + 4 + 1:] # Skip 35th band since it's a duplicate of 36
            msi_out = np.concatenate((msi_out_vis, msi_out_nir), axis=0).T

            msi_in = msi_out[:, :: 4, :: 4]
            np.save(path.join(dest_path, key), arr)



if __name__ == "__main__":


with h5py.File(
