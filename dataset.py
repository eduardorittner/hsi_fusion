import torch
from torch.utils.data import Dataset
import numpy as np
from glob import glob
from os import path
from typing import Optional, List, Tuple, Callable
import pywt
from pytorch_lightning import LightningDataModule
import pytorch_lightning as pl


class IcasspDataModule(LightningDataModule):
    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)

    def setup(self, stage=None):
        # TODO: Read transforms and shit from hparams
        base_path = self.hparams.data_path

        train_transform = None
        train_preprocessing = None
        res = 256
        self.train = IcasspDataset(
            path.join(base_path, "train"),
            train_transform,
            train_preprocessing,
            res,
            crop_strategy,
            bands,
            bands_strategy,
        )


class IcasspDataset(Dataset):
    def __init__(
        self,
        base_path: str,
        transform: Optional[Callable] = None,
        preprocessing: Optional[str] = None,
        fold: Optional[int] = None,
        res: Optional[int] = 256,
        crop_strategy: Optional[str] = "center",
        bands: Optional[int] = 31,
        bands_strategy: Optional[str] = None,
        **kwargs,
    ):
        super().__init__()

        for k, v in kwargs:
            print(f"WARNING: Ignoring dataset argument {k}: {v}")

        self.base_path = base_path
        self.transform = transform
        self.preprocessing = preprocessing
        self.fold = fold
        self.res = res
        self.crop_strategy = crop_strategy
        self.bands = bands
        self.bands_strategy = bands_strategy

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

    def preprocess_res(self, image: np.ndarray) -> np.ndarray:
        current_res = image.shape[0]
        if current_res == self.res:
            return image

        if current_res < self.res:
            return self.upsample_res(image, self.res)

        ratio = current_res // self.res
        assert ratio % 2 == 0, f"New resolution {self.res} must be a power of 2"

        return image[::ratio, ::ratio, :]

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

    def preprocess_bands(self, image: np.ndarray) -> np.ndarray:
        current_bands = image.shape[2]

        if current_bands == self.bands:
            return image

        if current_bands < self.bands:
            return self.upsample_band(image, self.bands)

        if self.bands_strategy is None:
            return image[:, :, 8:39]  # Considering blue starts at around 470nm

        else:
            assert False, "No other bands strategy implemented"

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

    def choose_center(self, image: np.ndarray) -> np.ndarray:
        res = image.shape[0]
        return image[res // 4 : res * 3 // 4, res // 4 : res * 3 // 4, :]

    def load_image(
        self, msi_in_path: str, rgb_in_path: str, msi_out_path: str
    ) -> Tuple[np.ndarray, np.ndarray]:
        # TODO: Use self.res and self.bands variables to load image correctly

        msi_in = np.load(msi_in_path)
        rgb_in = np.load(rgb_in_path)
        msi_out = np.load(msi_out_path)

        return rgb_in, msi_in, msi_out

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns (msi_in[1024,1024,61] + rgb_in[1024,1024,61]) + (msi_out[1024,1024,61])
        Resulting size is [122,1024,1024]
        """

        rgb_in, msi_in, msi_out = self.load_image(
            self.msi_in[idx], self.rgb_in[idx], self.msi_out[idx]
        )

        if self.transform is not None:
            rgb_in, msi_in, msi_out = self.transform(rgb_in, msi_in, msi_out)

        print(rgb_in.shape, msi_in.shape, msi_out.shape)

        exit(1)

        input = np.concatenate((msi_in, rgb_in), axis=2)

        input = torch.from_numpy(input).float()
        target = torch.from_numpy(msi_out).float()

        return input, target
