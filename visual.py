import glob
from os.path import join
import numpy as np
import argparse
import matplotlib
import matplotlib.pyplot as plt
from random import choice
from utils.image_id import image_id, mask_id
from typing import List


def load_mask(masks: List[str], id: str) -> str | None:
    for mask in masks:
        if mask_id(mask) == id:
            return mask

    print(f"[WARNING]: Mask id {id} was not found, output shown without mask")
    return None


def visualize(name: str, img: np.ndarray, band: int | None, mask: np.ndarray | None):
    if mask is not None:
        assert (
            img[:, :, 0].shape == mask.shape
        ), f"Image shape ({img[:,:,0].shape}) and mask shape ({mask.shape}) must be equal."

    if band is None:
        print(f"No band was provided, visualizing first band")
        if mask is not None:
            plt.imshow(img[:, :, 0] * mask)
        else:
            plt.imshow(img[:, :, 0])

        plt.show()
    elif 0 <= band <= 60:
        if mask is not None:
            plt.imshow(img[:, :, band] * mask)
        else:
            plt.imshow(img[:, :, band])
    elif band == 61:
        cols = 8
        rows = 8
        fig, axes = plt.subplots(rows, cols, figsize=(12, 12))
        for i in range(61):
            row, col = divmod(i, cols)
            if mask is not None:
                for i in range(img.shape[2]):
                    axes[row, col].imshow(img[:, :, i] * mask)
            else:
                for i in range(img.shape[2]):
                    axes[row, col].imshow(img[:, :, i] * mask)
            axes[row, col].set_xticks([])
            axes[row, col].set_yticks([])

        plt.tight_layout()
        plt.show()
    else:
        print(f"[ERROR]: Band ({band}) must be between 0 and 61")


msi_in_dir = "/home/eduardo/data/msi_in/"
hsi_in_dir = "/home/eduardo/data/hsi_in/"
hsi_out_dir = "/home/eduardo/data/hsi_out/"
mask_dir = "/home/eduardo/data/masks/"

if __name__ == "__main__":
    matplotlib.use("TkAgg")

    parser = argparse.ArgumentParser(
        prog="visualizer",
        description="Data visualizer",
    )

    parser.add_argument(
        "-i",
        "--image",
        type=str,
        help="Path of image to visualize, if None is provided a random image will be selected",
    )
    parser.add_argument(
        "-b",
        "--band",
        type=int,
        help="What band to visualize. If None is given shows all bands",
    )
    parser.add_argument(
        "-t",
        "--type",
        type=str,
        help="Type of image: msi_in, hsi_in, hsi_out, prediction. If none is provided hsi_out will be selected",
    )
    parser.add_argument("-m", "--masks", type=bool, help="Output with masks")

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

    if args.masks:
        mask = load_mask(sorted(glob.glob(join(mask_dir, "*.npy"))), id)
    else:
        mask = None

    if mask is not None:
        mask = np.load(mask)

    visualize(id, img, args.band, mask)
