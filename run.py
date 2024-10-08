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
    parser.add_argument(
        "-i",
        "--imageid",
        help="Which image(s) to load by id, if none are provided a random image will be chosen",
        default=None,
        action="append",
    )
    parser.add_argument(
        "-m",
        "--mode",
        help="'visual': visualize results, 'res': calculate results, 'store': store results in file, 'all': all of the previous",
        default="visual",
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

    # Disable randomness and such
    # model.eval()
    # model.freeze()

    if args.imageid is None:
        # Get random image
        image = choice(glob.glob(join(data_dir, "msi_in/*.npy")))
        id = image_id(image)
        args.imageid = [id]

    for id in args.imageid:
        print(f"Loading image {id}")

        msi_file = join(data_dir, f"msi_in/{id}.npy")

        if not isfile(join(msi_file)):
            print(f"[ERROR]: Couldn't load file '{msi_file}'")
            continue

        # Predict the image using the trained model
        early_stop = pl.callbacks.early_stopping.EarlyStopping(monitor="val_loss")
        trainer = pl.Trainer(devices=1, accelerator="gpu", callbacks=[early_stop])
        pred = trainer.predict(model, dataloader)

        # Inverse the coefficients
        shapes = pywt.wavedecn_shapes(
            (65, 256, 256), wavelet="db2", mode="periodization", level=2
        )
        coeffs = pywt.array_to_coeffs(pred.numpy(), shapes)
        pred_image = pywt.waverecn(coeffs, wavelet="db2", mode="periodization")
        output = None

        ssim = metric_ssim(output, pred_image)

        if args.mode in ("res", "store", "all"):
            sam = metric_sam(output, pred_image)
            psnr = metric_psnr(output, pred_image)

            print(f"Calculated metrics: SSIM {ssim[0]} | SAM {sam} | PSNR {psnr}")
            # TODO: save results in a file

        if args.mode in ("visual", "all"):
            # We use SSIM to present the least + most different bands

            min = ssim[1:].argmin()
            max = ssim[1:].argmax()

            # TODO: Confirm which axis contains the bands
            plt.imshow(pred_image[:, :, min + 1])
            plt.show()
            plt.imshow(output[:, :, min + 1])
            plt.show()
            plt.imshow(pred_image[:, :, max + 1])
            plt.show()
            plt.imshow(output[:, :, max + 1])
            plt.show()
