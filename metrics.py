from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import numpy as np
import torch


def metric_ssim(result, expected):
    """
    Returns the ssim (structural similarity index) of the result
    The first element is the ssim across all bands, and the subsequent elements
    are the ssim for each band
    """

    assert result.shape == expected.shape

    results = []

    results.append(
        ssim(result, expected, data_range=result.max() - result.min(), channel_axis=2)
    )

    for i in range(result.shape[2]):
        results.append(
            ssim(
                result[:, :, i],
                expected[:, :, i],
                data_range=result[:, :, i].max() - result[:, :, i].min(),
            )
        )

    return results


def metric_sam(pred, target):
    """Returns the sam of the result"""

    assert (
        pred.shape == target.shape
    ), f"Pred and Target must have equal shape, have {pred.shape} and {target.shape}"

    result = []

    for i in range(pred.shape[2]):
        result.append(sam(pred[:, :, i], target[:, :, i])[0])

    return torch.stack(result).mean()


def metric_psnr(pred, target):
    return psnr(pred, target, data_range=pred.max() - pred.min())
