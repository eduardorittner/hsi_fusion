from dataset import IcasspDataModule
import monai
import pywt
from models.unet import UNetModel
from networks.baseline import Baseline
from torch.utils.data import DataLoader
import torch
import torch.nn.functional as F
from utils import checkpoint, read_yaml
from transforms.get import get_transform
import pytorch_lightning as pl

if __name__ == "__main__":

    hparams = read_yaml.read_yaml("/home/eduardo/hsi_fusion/models/example.yaml", False)

    dataloader = IcasspDataModule(hparams)

    dataloader.setup()

    unet = monai.networks.nets.UNet(
        spatial_dims=2,
        in_channels=65,
        out_channels=63,
        channels=(2, 4, 8, 16),
        strides=(2, 2, 2),
    )

    torch.set_float32_matmul_precision("medium")

    def dwt(input):
        return torch.from_numpy(
            pywt.coeffs_to_array(
                pywt.wavedecn(input, "db4", level=2, mode="periodization"), padding=0.0
            )[0]
        )

    model = UNetModel(
        net=unet,
        loss=torch.nn.MSELoss(),
        learning_rate=1e-2,
        optimizer=torch.optim.AdamW,
        dwt=dwt,
    )

    early_stop = pl.callbacks.early_stopping.EarlyStopping(monitor="val_loss")

    trainer = pl.Trainer(devices=1, accelerator="gpu", callbacks=[early_stop])
    trainer.fit(model=model, datamodule=dataloader)
