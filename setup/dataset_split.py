# Split the dataset into train, validation and test sets

import argparse
import glob
from os.path import join, isdir, mkdir

def dataset_split(source_dir, dest_dir, move):
    """
    Split the dataset into 60% train, 20% val and 20% test
    """
    rgb_files = sorted(glob.glob(join(source_dir, "rgb_in")))
    msi_files = sorted(glob.glob(join(source_dir, "msi_in")))
    msi_out_files = sorted(glob.glob(join(source_dir, "msi_out")))

    assert(
        len(rgb_files) == len(msi_files) == len(msi_out_files),
        "All directories must contain the same number of files"
    )

    if not isdir(dest_dir):
        mkdir(dest_dir)

    if not isdir(join(dest_dir, "train")):
        mkdir(join(dest_dir, "train"))

    if not isdir(join(dest_dir, "val")):
        mkdir(join(dest_dir, "val"))

    if not isdir(join(dest_dir, "test")):
        mkdir(join(dest_dir, "test"))


    for f1, f2, f3 in zip(rgb_files, msi_files, msi_out_files):
    # TODO: finish
        pass

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Split dataset into train, validation and test sets"
    )

    parser.add_argument(
        "source_dir",
        type=str,
        help="Path to source directory",
        default=None,
    )


    parser.add_argument(
        "move", type=str, help="[y]: Move files, [n]: Copy files", default="n"
    )

    parser.add_argument(
        "dest_dir",
        type=str,
        help="Path to destination directory",
        default=None,
    )
