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

    def setup_nosplit(self):
        base_path = self.hparams.data_path

        files = (
            self.hparams.train_split + self.hparams.val_split + self.hparams.test_split
        )
        transform = get_transform(self.hparams.transform)
        self.all = IcasspDataset(base_path, files, transform, None)

    def setup(self, stage=None):
        base_path = self.hparams.data_path

        train_split = self.hparams.train_split
        val_split = self.hparams.val_split
        test_split = self.hparams.test_split
        train_transform = get_transform(self.hparams.transform)
        train_preprocessing = None  # Do we really need preprocessing?

        self.train = IcasspDataset(
            base_path,
            train_split,
            train_transform,
            train_preprocessing,
        )

        self.val = IcasspDataset(
            base_path,
            val_split,
            train_transform,
            train_preprocessing,
        )

        self.test = IcasspDataset(
            base_path,
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

    def all_dataloader(self):
        return DataLoader(
            self.all,
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
        self.hsi_out: List[str] = sorted(
            [
                i
                for i in glob(path.join(self.base_path, "hsi_out", fmt))
                if any(name in i for name in names)
            ]
        )
        self.total_files = len(self.hsi_in)

        dir_lens_match = (
            len(self.hsi_in)
            == len(self.msi_in)
            == len(self.hsi_out)
            == self.total_files
        )

        assert (
            dir_lens_match
        ), f"One of the 3 directories does not contain {self.total_files} files: msi_in - {len(self.msi_in)}, hsi_in - {len(self.hsi_in)}, hsi_out - {len(self.hsi_out)}"

        print(self.init_str())

    def init_str(self) -> str:
        return f"Initialized {self.total_files} images Dataset with preprocessing: {self.preprocessing} in {self.base_path}"

    def __len__(self) -> int:
        return self.total_files

    def load_image(
        self, hsi_in_path: str, msi_in_path: str, hsi_out_path: str
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

        hsi_in = np.load(hsi_in_path)
        msi_in = np.load(msi_in_path)
        hsi_out = np.load(hsi_out_path)

        return msi_in, hsi_in, hsi_out

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:

        msi_in, hsi_in, hsi_out = self.load_image(
            self.hsi_in[idx], self.msi_in[idx], self.hsi_out[idx]
        )

        msi_in = torch.from_numpy(msi_in)
        hsi_in = torch.from_numpy(hsi_in)
        hsi_out = torch.from_numpy(hsi_out)

        if self.transform is not None:
            msi_in, hsi_in, hsi_out = self.transform(msi_in, hsi_in, hsi_out)

        msi_in = torch.swapaxes(msi_in, 0, 2)
        hsi_in = torch.swapaxes(hsi_in, 0, 2)
        hsi_out = torch.swapaxes(hsi_out, 0, 2)

        input = torch.cat((hsi_in, msi_in), 0)
        target = hsi_out

        return input, target
