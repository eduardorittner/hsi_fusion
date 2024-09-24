from dwt.dwt import fuse_3dDWT, fuse_2dDWT
from dwt.average import fuse_average
import glob
from os.path import join
import argparse
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--source", type=str, help="Source directory")
    parser.add_argument("-d", "--dest", type=str, help="Destination directory")
    parser.add_argument(
        "-i", "--id", type=str, help="Which image to fuse based on id (without '.npy')"
    )

    args = parser.parse_args()

    if args.source is None or args.dest is None or args.id is None:
        print("[ERROR]: Please provide both arguments")
        exit(1)

    msi_in_files = sorted(glob.glob(join(args.source, "msi_in/*.npy")))
    hsi_in_files = sorted(glob.glob(join(args.source, "hsi_in/*.npy")))
    hsi_out_files = sorted(glob.glob(join(args.source, "hsi_out/*.npy")))

    msi_in = next(filter(lambda x: args.id in x, msi_in_files), None)
    hsi_in = next(filter(lambda x: args.id in x, hsi_in_files), None)
    hsi_out = next(filter(lambda x: args.id in x, hsi_out_files), None)

    if msi_in and hsi_in and hsi_out:
        msi_in = np.load(msi_in)
        hsi_in = np.load(hsi_in)
        hsi_out = np.load(hsi_out)

    else:
        print(f"[ERROR]: Couldn't find file: {args.id}")
        exit(1)

    result = fuse_3dDWT(msi_in, hsi_in, "db1", 2, None)

    np.save(join(args.dest, "result.npy"), result)

    plt.imshow(result[:, :, 20])
    plt.show()
