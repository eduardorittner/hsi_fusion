import numpy as np
from typing import List, Callable


def fuse_average(
    rgb_in: np.ndarray, msi_in: np.ndarray, transform: Callable | None
) -> np.ndarray:

    if transform is not None:
        rgb_in, msi_in, _ = transform(rgb_in, msi_in, None)

    return (rgb_in + msi_in) / 2
