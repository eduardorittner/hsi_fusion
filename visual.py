import argparse
import numpy as np
import glob
from os.path import join
import matplotlib.pyplot as plt
from random import choice
from utils.image_id import image_id


def visualize_band(name: str, img: np.ndarray, band: int):
    plt.imshow(img[:, :, band])


def visualize_bands_all(name: str, img: np.ndarray):
    for i in range(img.shape[2]):
        plt.imshow(img[:, :, i])


msi_in_dir = "/home/eduardo/data/msi_in/"
hsi_in_dir = "/home/eduardo/data/hsi_in/"
hsi_out_dir = "/home/eduardo/data/hsi_out/"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="visualizer",
        description="Data visualizer",
    )

    parser.add_argument(
        "i",
        "--image",
        type=str,
        help="Path of image to visualize, if None is provided a random image will be selected",
    )
    parser.add_argument(
        "t",
        "--type",
        type=str,
        help="Type of image: msi_in, hsi_in, hsi_out, prediction. If none is provided hsi_out will be selected",
    )

    args = parser.parse_args()

    if args.image is not None:
        id = image_id(args.image)
        img = np.load(args.image)
    elif args.type is not None:
        match args.type:
            case "msi_in":
                img = choice(sorted(glob.glob(join(msi_in_dir, "*.npy"))))
            case "hsi_in":
                img = choice(sorted(glob.glob(join(hsi_in_dir, "*.npy"))))
            case "hsi_out":
                img = choice(sorted(glob.glob(join(hsi_out_dir, "*.npy"))))
            case "prediction":
                print("[ERROR]: prediction visualization not implemented yet")
                exit(1)
            case _:
                print(f"[ERROR]: Unknown type: {args.type}")
                exit(1)

        id = image_id(img)
        img = np.load(img)

    else:
        hsi_out_files = sorted(glob.glob(join(hsi_out_dir, "*.npy")))
        img = choice(hsi_out_files)
        print(f"Visualizing {img}")
        id = image_id(img)
        img = np.load(img)

    visualize_band(id, img, 9)
