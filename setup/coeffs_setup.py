import numpy as np
import glob
import time
from os import path
import pywt
from tqdm import tqdm


def name(file):
    return file.split("/")[-1].split(".")[0]


base_path = "/home/eduardo/data/hsi_out/"
dest_path = "/home/eduardo/data/coeffs/"

files = sorted(glob.glob(path.join(base_path, "*.npy")))

for i in tqdm(range(len(files))):
    file = files[i]
    img = np.load(file)

    img = img[::4, ::4, :]  # Resolução de 256x256

    coeffs, slices = pywt.coeffs_to_array(pywt.wavedecn(img, "db2", level=2))

    np.save(path.join(dest_path, name(file)), coeffs)
