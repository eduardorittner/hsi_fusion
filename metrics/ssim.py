from skimage.metrics import structural_similarity as ssim
import numpy as np
from typing import List, Dict


def metric_ssim(result: np.ndarray, expected: np.ndarray) -> NDArray:
    """
    Returns the ssim (structural similarity index) of the result
    The first element is the ssim across all bands, and the subsequent elements
    are the ssim for each band
    """

    assert result.shape == expected.shape

    results = np.empty((62))

    results[0] = ssim(
        result, expected, data_range=result.max() - result.min(), channel_axis=2
    )

    for i in range(result.shape[2]):
        results[i + 1] = ssim(
            result[:, :, i],
            expected[:, :, i],
            data_range=result[:, :, i].max() - result[:, :, i].min(),
        )

    return results


def metric_join_ssim(results: Dict) -> np.ndarray:
    r = np.zeros((62))

    for value in results.values():
        r += value

    r /= len(results.keys())

    return r
