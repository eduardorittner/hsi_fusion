from os.path import join, isfile
import monai
from transforms.get import get_transform
import glob
import yaml
import pywt
import matplotlib.pyplot as plt
import pytorch_lightning as pl
import torch
from models.unet import UNetModel, UnetUpsample
import argparse
from random import choice
from utils.image_id import image_id
from utils.read_yaml import read_yaml
import numpy as np
from dataset import IcasspDataModule
from metrics.ssim import metric_ssim
from metrics.sam import metric_sam
from metrics.psnr import metric_psnr


data_dir = "/home/eduardo/data"


def dwt(input):
    return torch.from_numpy(
        pywt.coeffs_to_array(
            pywt.wavedecn(input, "db4", level=2, mode="periodization"), padding=0.0
        )[0]
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c", "--checkpoint", type=str, help="Path to saved checkpoint", required=True
    )

    args = parser.parse_args()

    hparams = read_yaml("/home/eduardo/hsi_fusion/models/example.yaml", False)

    dataloader = IcasspDataModule(hparams)

    dataloader.setup_nosplit()

    unet = monai.networks.nets.UNet(
        spatial_dims=2,
        in_channels=65,
        out_channels=63,
        channels=(2, 4, 8, 16),
        strides=(2, 2, 2),
    )

    model = UNetModel.load_from_checkpoint(
        args.checkpoint, net=unet, loss=torch.nn.MSELoss
    )

    # Predict the image using the trained model
    early_stop = pl.callbacks.early_stopping.EarlyStopping(monitor="val_loss")
    trainer = pl.Trainer(devices=1, accelerator="gpu", callbacks=[early_stop])
    results = trainer.predict(model, dataloader)

    for result in results:
        pred, target, idx = result

        pred = pred.squeeze().numpy(force=True)
        target = target.squeeze().numpy(force=True)

        np.save(join(data_dir, "pred", "pred" + str(idx)), pred)
        np.save(join(data_dir, "pred", "target" + str(idx)), target)
