import numpy as np
import glob
from os import path
import pywt

base_path = "/home/eduardo/data/msi_out/"

files = sorted(glob.glob(path.join(base_path, "*.npy")))

for file in files:
    img = np.load(file)
    coeffs, slices, shapes = pywt.ravel_coeffs(pywt.wavedecn(img, "coif1", level=2))
    print(coeffs.max())
    print(coeffs.min())
    exit()
