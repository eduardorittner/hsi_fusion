import h5py
from PIL import Image
import numpy as np
import os
import matplotlib.pyplot as plt

# Utility for reading from a .mat or .jpg into a numpy array transparently

jpg_shape = (3, 482, 512)
mat_shape = (31, 482, 512)


def arad_open(file: str) -> np.ndarray:
    if os.path.isfile(file):
        if ".mat" in file:
            img = np.asarray(h5py.File(file)["cube"])
            img = img.swapaxes(1, 2)

            assert (
                img.shape == mat_shape
            ), f".mat images should have shape: {mat_shape}, found: {img.shape}"

            return img
        elif ".jpg" in file:
            img = np.asarray(Image.open(file))
            img = img.swapaxes(0, 1)
            img = img.swapaxes(0, 2)

            assert (
                img.shape == jpg_shape
            ), f".jpg images should have shape: {jpg_shape}, found: {img.shape}"

            return img
        raise Exception(f"[ERROR]: expected .mat or .jpg file, got: {file}")
    raise Exception(f"[ERROR]: {file} is not a file.")
