import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from glob import glob
from os import path
from typing import Optional, List, Tuple, Callable
import pywt
from pytorch_lightning import LightningDataModule
import pytorch_lightning as pl
from transforms.get import get_transform


class IcasspDataModule(LightningDataModule):
    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)

    def setup(self, stage=None):
        base_path = self.hparams.data_path

        train_transform = get_transform(self.hparams.transform)
        train_preprocessing = None  # Do we really need preprocessing?
        res = self.hparams.res
        bands = self.hparams.bands

        self.train = IcasspDataset(
            path.join(base_path, "train"),
            train_transform,
            train_preprocessing,
            res,
            bands,
        )

        self.val = IcasspDataset(
            path.join(base_path, "val"),
            train_transform,
            train_preprocessing,
            res,
            bands,
        )

        self.test = IcasspDataset(
            path.join(base_path, "test"),
            train_transform,
            train_preprocessing,
            res,
            bands,
        )

    def train_dataloader(self):
        return DataLoader(
            self.train,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.nworkers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.nworkers,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.nworkers,
        )


class IcasspDataset(Dataset):
    def __init__(
        self,
        base_path: str,
        transform: Optional[Callable] = None,
        preprocessing: Optional[str] = None,
        fold: Optional[int] = None,
        res: Optional[int] = 256,
        bands: Optional[int] = 31,
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
        self.bands = bands

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
