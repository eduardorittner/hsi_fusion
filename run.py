from os.path import join, isfile
import monai
from transforms.get import get_transform
import glob
import yaml
import pywt
import matplotlib.pyplot as plt
import torch
from models.unet import UNetModel
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

    dataloader.setup()

    unet = monai.networks.nets.UNet(
        spatial_dims=2,
        in_channels=65,
        out_channels=74,
        channels=(2, 4, 8, 16),
        strides=(2, 2, 2),
    )

    upsample = monai.networks.blocks.Upsample(
        spatial_dims=2,
        in_channels=74,
        out_channels=74,
        size=(269, 269),
        mode="nontrainable",
        interp_mode="bilinear",
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

    torch.set_float32_matmul_precision("medium")

    model = UnetUpsample(unet, upsample)

    def dwt(input):
        return torch.from_numpy(
            pywt.coeffs_to_array(pywt.wavedecn(input, "db4", level=2), padding=0.0)[0]
        )

    model = UNetModel(
        net=model,
        loss=monai.losses.ssim_loss.SSIMLoss(spatial_dims=2),
        learning_rate=1e-2,
        optimizer=torch.optim.AdamW,
        dwt=dwt,
    )

    model = UNetModel.load_from_checkpoint(
        args.checkpoint,
        net=model,
        loss=monai.losses.ssim_loss.SSIMLoss(spatial_dims=2),
        learning_rate=1e-2,
        optimizer=torch.optim.AdamW,
        dwt=dwt,
    )
    # Disable randomness and such
    model.eval()

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

        msi_in = torch.from_numpy(np.load(msi_file))
        hsi_in = torch.from_numpy(np.load(join(data_dir, f"hsi_in/{id}.npy")))
        hsi_out = torch.from_numpy(np.load(join(data_dir, f"hsi_out/{id}.npy")))

        transform = get_transform("256x256")
        msi_in, hsi_in, hsi_out = transform(msi_in, hsi_in, hsi_out)

        msi_in = torch.swapaxes(msi_in, 0, 2)
        hsi_in = torch.swapaxes(hsi_in, 0, 2)
        hsi_out = torch.swapaxes(hsi_out, 0, 2)

        input = torch.cat((hsi_in, msi_in), 0)
        output = hsi_out

        # Predict the image using the trained model
        pred = model(input)

        # Inverse the coefficients
        shapes = pywt.wavedecn_shapes(
            (65, 256, 256), wavelet="db2", mode="periodization", level=2
        )
        coeffs = pywt.array_to_coeffs(pred.numpy(), shapes)
        pred_image = pywt.waverecn(coeffs, wavelet="db2", mode="periodization")
        output = output.numpy()

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
