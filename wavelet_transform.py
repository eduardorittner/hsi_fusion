# Wavelet transform to be applied to the images in the dataset

import numpy as np
import pywt
from typing import Callable, Tuple


def create_dwt_fn(
    wavelet: str, level: int
) -> Callable[[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    """
    Returns a function that applies the given transform to an input and target
    to be used inside the IcasspDataset class
    """

    def dwt(input: np.ndarray, target: np.ndarray) -> [np.ndarray, np.ndarray]:
        input_1, _ = pywt.coeffs_to_array(
            pywt.wavedecn(input[:, :, :1024], wavelet=wavelet, level=level)
        )
        input_2, _ = pywt.coeffs_to_array(
            pywt.wavedecn(input[:, :, 1024:], wavelet=wavelet, level=level)
        )
        input = np.concatenate((input_1, input_2), axis=2)

        target, _ = pywt.coeffs_to_array(
            pywt.wavedecn(target, wavelet=wavelet, level=level)
        )

        return input, target

    return dwt
