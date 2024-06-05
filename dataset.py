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

        train_split = self.hparams.train_split
        val_split = self.hparams.val_split
        test_split = self.hparams.test_split
        train_transform = get_transform(self.hparams.transform)
        train_preprocessing = None  # Do we really need preprocessing?

        self.train = IcasspDataset(
            path.join(base_path, "train"),
            train_split,
            train_transform,
            train_preprocessing,
        )

        self.val = IcasspDataset(
            path.join(base_path, "val"),
            val_split,
            train_transform,
            train_preprocessing,
        )

        self.test = IcasspDataset(
            path.join(base_path, "test"),
            test_split,
            train_transform,
            train_preprocessing,
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
        names: List[str],
        transform: Optional[Callable] = None,
        preprocessing: Optional[str] = None,
        fold: Optional[int] = None,
        **kwargs,
    ):
        super().__init__()

        for k, v in kwargs:
            print(f"WARNING: Ignoring dataset argument {k}: {v}")

        self.base_path = base_path
        self.names = names
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

        self.hsi_in: List[str] = sorted(
            [
                i
                for i in glob(path.join(self.base_path, "hsi_in", fmt))
                if any(name in i for name in names)
            ]
        )
        self.msi_in: List[str] = sorted(
            [
                i
                for i in glob(path.join(self.base_path, "msi_in", fmt))
                if any(name in i for name in names)
            ]
        )
        self.coeffs_out: List[str] = sorted(
            [
                i
                for i in glob(path.join(self.base_path, "coeffs_out", fmt))
                if any(name in i for name in names)
            ]
        )
        self.total_files = len(self.hsi_in)

        dir_lens_match = (
            len(self.hsi_in)
            == len(self.msi_in)
            == len(self.coeffs_out)
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
        self, hsi_in_path: str, msi_in_path: str, coeffs_out_path: str
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

        hsi_in = np.load(hsi_in_path)
        msi_in = np.load(msi_in_path)
        coeffs_out = np.load(coeffs_out_path)

        return msi_in, hsi_in, coeffs_out

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns (hsi_in[1024,1024,61] + msi_in[1024,1024,61]) + (coeffs_out[1024,1024,61])
        Resulting size is [122,1024,1024]
        """

        msi_in, hsi_in, coeffs_out = self.load_image(
            self.hsi_in[idx], self.msi_in[idx], self.coeffs_out[idx]
        )

        msi_in = torch.from_numpy(msi_in)
        hsi_in = torch.from_numpy(hsi_in)
        coeffs_out = torch.from_numpy(coeffs_out)

        if self.transform is not None:
            msi_in, hsi_in = self.transform(msi_in, hsi_in, None)

        input = torch.cat((hsi_in, msi_in), 2)
        target = coeffs_out

        return input, target
