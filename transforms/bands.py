import torch
import numpy as np
from typing import Tuple, Optional, List


class Bands:
    def __init__(self, low_bands: int, high_bands: int, bands_strategy: str):
        self.low_bands = low_bands
        self.high_bands = high_bands
        self.bands_strategy = None

    def upsample_band(
        self,
        image: np.ndarray,
        new_bands: int,
        band_index: Optional[List[int]] = [7, 14, 30, 31],
    ) -> np.ndarray:
        """
        Upsamples the given image to have number of bands equal to new_bands
        Expectes images of shape [spatial][spatial][spectral]
        """

        # band_index is a list of the current image's bands' corresponding
        # indexes in the resulting image. This is done for when the band
        # frequencies are not evenly spaced

        result_shape = (image.shape[0], image.shape[1], new_bands)
        result = np.empty(result_shape)

        assert (
            new_bands > image.shape[2]
        ), "Number of bands must be higher than the original picture"
        assert len(band_index) == image.shape[2]
        "Number of band indexes must be equal to number of bands in the image"

        previous = -1
        next = 0
        cutoff = 0

        for i in range(new_bands):
            if next < len(band_index) and i > band_index[next]:
                next += 1
                previous += 1
                if next == len(band_index):
                    cutoff = new_bands
                else:
                    cutoff = (
                        band_index[previous]
                        + (band_index[next] - band_index[previous]) // 2
                    )

            if i >= cutoff:
                result[:, :, i] = image[:, :, next]

            else:
                result[:, :, i] = image[:, :, previous]

        return result

    def preprocess_bands(self, image: np.ndarray, bands: int) -> np.ndarray:
        current_bands = image.shape[2]

        if current_bands == self.high_bands:
            return image

        if current_bands < self.high_bands:
            return self.upsample_band(image, self.high_bands)

        if self.bands_strategy is None:
            # Considering blue starts at around 470nm
            return image[:, :, 8 : 8 + self.high_bands]
        else:
            assert False, "No other bands strategy implemented"

    def __call__(
        self, x: np.ndarray, y: np.ndarray, z: Optional[np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        if z is not None:
            return (
                self.preprocess_bands(x, self.low_bands),
                self.preprocess_bands(y, self.high_bands),
                self.preprocess_bands(z, self.high_bands),
            )
        else:
            return (
                self.preprocess_bands(x, self.low_bands),
                self.preprocess_bands(y, self.high_bands),
                None,
            )

    def __str__(self) -> str:
        return f"High bands: {high_bands}, Low bands: {low_bands}"
