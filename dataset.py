import torch
from torch.utils.data import Dataset
import numpy as np
from glob import glob
from os import path
from typing import Optional, List, Tuple, Callable
import pywt


class IcasspDataset(Dataset):
    def __init__(
        self,
        base_path: str,
        transform: Optional[Callable] = None,
        preprocessing: Optional[str] = None,
        fold: Optional[int] = None,
        **kwargs,
    ):
        super().__init__()

        for k, v in kwargs:
            print(f"WARNING: Ignoring dataset argument {k}: {v}")

        self.base_path = base_path
        self.transform = transform
        self.preprocessing = preprocessing
        self.fold = fold

        if self.preprocessing is None:
            fmt = "*.npy"
            is_coefficient = False
        elif self.preprocessing == "dwt":
            fmt = "*.npy"
            is_coefficient = True

        self.fmt = fmt
        self.is_coefficient = is_coefficient

        self.msi_in: List[str] = sorted(
            glob(path.join(self.base_path, "downsampled", fmt))
        )
        self.rgb_in: List[str] = sorted(glob(path.join(self.base_path, "rgb-nir", fmt)))
        self.msi_out: List[str] = sorted(glob(path.join(self.base_path, "msi", fmt)))
        self.total_files = len(self.msi_in)

        dir_lens_match = (
            len(self.msi_in)
            == len(self.rgb_in)
            == len(self.msi_out)
            == self.total_files
        )

        assert (
            dir_lens_match
        ), f"One of the 3 directories does not contain {self.total_files} files"

        print(self.init_str())

    def init_str(self) -> str:
        return f"Initialized {self.total_files} images Dataset with preprocessing: {self.preprocessing} in {self.base_path}"

    def __len__(self) -> int:
        return self.total_files

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

    def upsample_band(
        self, image: np.ndarray, new_bands: int, band_index: List[int] = [7, 14, 30, 31]
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

    def load_image(
        self, msi_in_path: str, rgb_in_path: str, msi_out_path: str
    ) -> Tuple[np.ndarray, np.ndarray]:
        msi_in = np.load(msi_in_path)
        rgb_in = np.load(rgb_in_path)
        msi_out = np.load(msi_out_path).transpose(2, 0, 1)

        msi_in = self.upsample_res(msi_in, rgb_in.shape[0])
        rgb_in = self.upsample_band(rgb_in, msi_in.shape[2])

        result = np.concatenate((msi_in, rgb_in), axis=2)

        return result, msi_out

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns (msi_in[1024,1024,61] + rgb_in[1024,1024,61]) + (msi_out[1024,1024,61])
        Resulting size is [122,1024,1024]
        """

        input, target = self.load_image(
            self.msi_in[idx], self.rgb_in[idx], self.msi_out[idx]
        )

        if self.transform is not None:
            input, target = transform(input, target)

        input = torch.from_numpy(input).float()
        target = torch.from_numpy(target).float()

        return input, target
