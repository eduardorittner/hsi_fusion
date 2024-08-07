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
import os
from tqdm import tqdm


def save2np(file, dest_path):
    msi_in_path = path.join(dest_path, "msi_in")
    hsi_in_path = path.join(dest_path, "hsi_in")
    hsi_out_path = path.join(dest_path, "hsi_out")

    if not path.isdir(msi_in_path):
        os.mkdir(msi_in_path)

    if not path.isdir(hsi_in_path):
        os.mkdir(hsi_in_path)

    if not path.isdir(hsi_out_path):
        os.mkdir(hsi_out_path)

    with h5py.File(file) as f:
        keys = list(f.keys())

        for key in tqdm(keys):
            arr = f[key][()]
            msi_in = arr[:4].T
            hsi_out_vis = arr[4:35]
            hsi_out_nir = arr[36:]  # Skip 35th band since it's a duplicate of 36
            hsi_out = np.concatenate((hsi_out_vis, hsi_out_nir), axis=0).T

            hsi_in = hsi_out[::4, ::4, :]

            np.save(path.join(msi_in_path, key), msi_in)
            np.save(path.join(hsi_in_path, key), hsi_in)
            np.save(path.join(hsi_out_path, key), hsi_out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Setup initial dataset with different files for msi_in, hsi_in and hsi_out from .hdf file"
    )

    parser.add_argument(
        "train_source_file",
        type=str,
        help="Path to train .hdf file",
        default=None,
    )

    parser.add_argument(
        "val_source_file",
        type=str,
        help="Path to val.hdf file",
        default=None,
    )

    parser.add_argument(
        "dest_dir",
        type=str,
        help="Path to directories where dataset will be stored",
        default=None,
    )

    args = parser.parse_args()

    if args.train_source_file is not None:
        print(
            f"Loading files from train file: {args.train_source_file} into: {args.dest_dir}"
        )
        save2np(args.train_source_file, args.dest_dir)

    if args.val_source_file is not None:
        print(
            f"Loading files from val file: {args.val_source_file} into: {args.dest_dir}"
        )
        save2np(args.val_source_file, args.dest_dir)
