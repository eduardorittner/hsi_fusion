import numpy as np
from typing import Tuple, List


class Crop:
    def __init__(self, high_size: int, high_offset: int, low_size, low_offset):
        self.low_size = low_size
        self.low_offset = low_offset
        self.high_size = high_size
        self.high_offset = high_offset

    def __call__(
        self, x: np.ndarray, y: np.ndarray, z: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

        high_stop = self.high_offset + self.high_size
        low_stop = self.low_offset + self.low_size

        return (
            x[self.high_offset : high_stop, self.high_offset : high_stop, :],
            y[self.low_offset : low_stop, self.low_offset : low_stop, :],
            z[self.high_offset : high_stop, self.high_offset : high_stop, :],
        )

    def __str__(self) -> str:
        return f"Crop of high_size: {self.high_size}, high_offset: {self.high_offset}, low_size: {self.low_size}, low_offset: {self.low_offset}"
