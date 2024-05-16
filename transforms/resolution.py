import torch
import numpy as np
from typing import Tuple, Optional


class Resolution:
    def __init__(self, low_res: int, high_res: int):
        self.low_res = low_res
        self.high_res = high_res

    def upsample_res(self, image: np.ndarray, new_res: int) -> np.ndarray:
        """
        Upscales the given image to new_res value
        expects image of shape [spatial][spatial][spectral]
        """

        res = (image.shape[0], image.shape[1])
        ratio = new_res // res[0]

        assert ratio > 1, "New res must be at least 2 times bigger than current res"

        result_type = image.dtype
        result_shape = (new_res, new_res, image.shape[2])
        result = np.empty(result_shape, dtype=result_type)

        for i in range(ratio):
            for j in range(ratio):
                result[i::ratio, j::ratio, :] = image

        return result

    def preprocess_res(self, image: np.ndarray, res: int) -> np.ndarray:
        current_res = image.shape[0]
        if current_res == res:
            return image

        if current_res < res:
            return self.upsample_res(image, res)

        ratio = current_res // res
        assert ratio % 2 == 0, f"New resolution {self.res} must be a power of 2"

        return image[::ratio, ::ratio, :]

    def __call__(
        self, x: np.ndarray, y: np.ndarray, z: Optional[np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        if z is not None:
            return (
                self.preprocess_res(x, self.high_res),
                self.preprocess_res(y, self.low_res),
                self.preprocess_res(z, self.high_res),
            )
        else:
            return (
                self.preprocess_res(x, self.high_res),
                self.preprocess_res(y, self.low_res),
                None,
            )

    def __str__(self) -> str:
        return f"High resolution: {high_res}, Low resolution: {low_res}"
