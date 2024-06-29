import numpy as np
from typing import List, Callable


def fuse_average(
    msi_in: np.ndarray, hsi_in: np.ndarray, transform: Callable | None
) -> np.ndarray:

    if transform is not None:
        msi_in, hsi_in, _ = transform(msi_in, hsi_in, None)

    return (msi_in + hsi_in) / 2
