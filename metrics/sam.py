import torch
import numpy as np
from typing import Dict


def sam(pred, target):
    pred = torch.from_numpy(pred)
    target = torch.from_numpy(target)
    pred, target = pred.squeeze(), target.squeeze()
    up = torch.sum((target * pred), dim=0)  # [w, h]
    down1 = torch.sum((target**2), dim=0).sqrt()
    down2 = torch.sum((pred**2), dim=0).sqrt()

    map = torch.arccos(up / (down1 * down2))
    score = torch.mean(map[~torch.isnan(map)])
    map[torch.isnan(map)] = 0
    return score, map


def metric_sam(pred: np.ndarray, target: np.ndarray) -> float:
    """Returns the sam of the result"""

    assert (
        pred.shape == target.shape
    ), f"Pred and Target must have equal shape, have {pred.shape} and {target.shape}"

    result = []

    for i in range(pred.shape[2]):
        result.append(sam(pred[:, :, i], target[:, :, i])[0])

    return torch.stack(result).mean()


def metric_join_sam(results: Dict) -> float:
    r = 0

    for value in results.values():
        r += value

    r /= len(results)

    return r
