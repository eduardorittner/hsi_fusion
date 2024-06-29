from skimage.metrics import peak_signal_noise_ratio as psnr
import numpy as np
from typing import Dict


def metric_psnr(target, pred):
    return psnr(target, pred, data_range=pred.max() - pred.min())


def metric_join_psnr(results: Dict) -> float:
    r = 0

    for value in results.values():
        r += value

    r /= len(results)

    return r
