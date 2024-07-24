from dataset import IcasspDataModule
import monai
import pywt
from models.basic import UNetModel
from networks.baseline import Baseline
from torch.utils.data import DataLoader
import torch
import torch.nn.functional as F
from utils import checkpoint, read_yaml
from transforms.get import get_transform
import pytorch_lightning as pl

hparams = read_yaml.read_yaml("/home/eduardo/hsi_fusion/models/example.yaml", False)

dataloader = IcasspDataModule(hparams)

dataloader.setup()

unet = monai.networks.nets.UNet(
    spatial_dims=2,
    in_channels=65,
    out_channels=74,
    channels=(2, 4, 8, 16),
    strides=(2, 2, 2),
)

upsample = monai.networks.nets.Upsampling(
    spatial_dims=2,
    in_channels=74,
    out_channels=74,
    scale_factor=1.05,
    kernel_size=3,
    mode="deconv",
)


class UnetUpsample(torch.nn.Module):
    def __init__(self, unet, upsample):
        super().__init__()
        self.unet = unet
        self.up = upsample

    def forward(self, x):
        x = self.unet(x)
        x = self.up(x)
        return x


model = UnetUpsample(unet, upsample)


def dwt(input):
    return torch.from_numpy(
        pywt.coeffs_to_array(pywt.wavedecn(input, "db4", level=2), padding=0.0)[0]
    )


model = UNetModel(
    net=model,
    loss=torch.linalg.lstsq,
    learning_rate=1e-2,
    optimizer=torch.optim.AdamW,
    dwt=dwt,
)

early_stop = pl.callbacks.early_stopping.EarlyStopping(monitor="val_loss")

trainer = pl.Trainer(devices=1, accelerator="gpu", callbacks=[early_stop])
trainer.fit(model=model, datamodule=dataloader)
