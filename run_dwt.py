from dwt.dwt import fuse_3dDWT
import argparse
import json
from os.path import join, isdir
from os import mkdir
from tqdm import trange
import glob
from typing import Dict, List, Callable
from transforms.get import get_transform
from transforms.resolution import Resolution
from transforms.bands import Bands
from transforms.compose import Compose
from metrics.ssim import metric_ssim, metric_join_ssim
from metrics.sam import metric_sam, metric_join_sam
from metrics.psnr import metric_psnr, metric_join_psnr
import numpy as np
from utils.image_id import image_id
from utils.read_yaml import read_yaml
from datetime import datetime


def run_dwt(
    rgb_in_files: List[str],
    msi_in_files: List[str],
    msi_out_files: List[str],
    method: str,
    wavelet: type[List[str] | str],
    level: int,
    metrics: List[str],
    dir: str,
    transforms: Callable,
):

    if isinstance(wavelet, str):
        wavelet_str = wavelet
        print("wa")
    else:
        wavelet_str = ""
        for wav in wavelet:
            wavelet_str += wav
            wavelet_str += "-"
        wavelet_str = wavelet_str[:-1]

    dir = dir + wavelet_str

    print(dir)
    mkdir(dir)

    with open(join(dir, "method.txt"), "w") as f:
        f.write(
            f"""
Reading files from: {rgb_in_files[0].split("/")[-3]}
dwt with wavelet(s): {wavelet}, level: {level}
metric(s): {metrics} stored in {dir}
"""
        )

    print(
        f"""
Reading files from: {rgb_in_files[0].split("/")[-3]}
dwt with wavelet(s): {wavelet}, level: {level}
metric(s): {metrics} stored in {dir}
    """
    )

    for i in trange(len(rgb_in_files)):
        rgb_in = np.load(rgb_in_files[i])
        msi_in = np.load(msi_in_files[i])
        expected = np.load(msi_out_files[i])

        rgb_in, msi_in, expected = transforms(rgb_in, msi_in, expected)

        result = fuse_3dDWT(rgb_in, msi_in, wavelet, level, transforms)
        id = image_id(rgb_in_files[i])

        results = {}

        if "ssim" in metrics:
            results["ssim"] = metric_ssim(result, expected)
        if "sam" in metrics:
            results["sam"] = metric_sam(result, expected)
        if "psnr" in metrics:
            results["psnr"] = metric_psnr(result, expected)

        save_image_result(id, results, dir)


def save_image_result(image_id: str, results: Dict, dir: str):
    with open(join(dir, image_id + ".json"), "w") as f:
        json.dump(results, f)


def aggregate_results(dir: str):
    results = {}
    files = sorted(glob.glob(dir + "*.json"))
    for file in files:
        id = image_id(file)
        results[id] = json.load(file)

    return results


def save_results(results: Dict, dir: str):
    date = datetime.today().strftime("%Y-%m-%d-%s")
    with open(join(dir, "results" + date + ".txt"), "w") as f:
        f.write(
            f"""Method: {results['method']}
Wavelet: {results['wavelet']}
Level: {results['level']}
-----------------------\n"""
        )

        if "ssim" in results.keys():
            metrics = results["ssim"]
            f.write("SSIM\n")
            f.write(f"Average: {metrics[0]}\n")
            for i in range(1, len(metrics)):
                f.write(f"Band {i-1}: {metrics[i]}\n")
            f.write("-----------------------\n")

        if "sam" in results.keys():
            f.write("SAM\n")
            f.write(f"{results['sam']}\n")
            f.write("-----------------------\n")

        if "psnr" in results.keys():
            f.write("PSNR\n")
            f.write(f"{results['psnr']}\n")
            f.write("-----------------------")


def run_dwt_suite(dir: str):
    config_files = sorted(glob.glob(join(dir, "*.yaml")))
    for file in config_files:
        config = read_yaml(file, False)
        print(f"Running {file}")
        rgb_in_files = sorted(glob.glob(join(config["rgb_in_files"], "*.npy")))
        msi_in_files = sorted(glob.glob(join(config["msi_in_files"], "*.npy")))
        msi_out_files = sorted(glob.glob(join(config["msi_out_files"], "*.npy")))
        results = run_dwt(
            rgb_in_files,
            msi_in_files,
            msi_out_files,
            config["method"],
            config["wavelet"],
            config["level"],
            config["metrics"].split(","),
            config["dir"],
            get_transform(config["transforms"]),
        )
        print("Results calculated:")
        if "ssim" in results.keys():
            print(f"SSIM: {results['ssim'][0]}")
        if "sam" in results.keys():
            print(f"sam: {results['sam']}")
        if "psnr" in results.keys():
            print(f"psnr: {results['psnr']}")
        print("------------------")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="fusion-runner",
        description="Runs the given fusion method and calculates the given metrics",
    )

    parser.add_argument("-o", "--source", type=str, help="Source directory")
    parser.add_argument("-m", "--method", type=str, help="Fusion method")
    parser.add_argument("-w", "--wavelet", type=str, help="Wavelet")
    parser.add_argument("-l", "--level", type=int, help="dwt level")
    parser.add_argument("-d", "--dir", type=str, help="Where results are stored")
    parser.add_argument("-e", "--metrics", type=str, help="Which metrics to compute")
    parser.add_argument("-c", "--config", type=str, help="Path to config")
    parser.add_argument("-s", "--suite", type=str, help="Path to configs")

    args = parser.parse_args()

    if args.suite is not None:
        run_dwt_suite(args.suite)
        exit(0)

    if args.config is not None:
        # Read from config file
        config = read_yaml(args.config, False)
        rgb_in_files = sorted(glob.glob(join(config["rgb_in_files"], "*.npy")))
        msi_in_files = sorted(glob.glob(join(config["msi_in_files"], "*.npy")))
        msi_out_files = sorted(glob.glob(join(config["msi_out_files"], "*.npy")))

        results = run_dwt(
            rgb_in_files,
            msi_in_files,
            msi_out_files,
            config["method"],
            config["wavelet"].split(","),
            config["level"],
            config["metrics"].split(","),
            config["dir"],
            get_transform(config["transforms"]),
        )

    else:
        if not isdir(args.source):
            print(f"ERROR: Directory {args.source} does not exist.")
            exit(1)

        rgb_in_files = sorted(glob.glob(join(args.source, "rgb_in/*.npy")))
        msi_in_files = sorted(glob.glob(join(args.source, "msi_in/*.npy")))
        msi_out_files = sorted(glob.glob(join(args.source, "msi_out/*.npy")))

        method = args.method

        if method != "3d-dwt":
            print(f"ERROR: {method} not implemented")
            exit(1)

        if args.wavelet is not None:
            wavelet = args.wavelet.split(",")
        else:
            print("ERROR: Must provide wavelet")
            exit(1)
        level = args.level
        dir = args.dir

        if args.metrics is not None:
            metrics = args.metrics.split(",")
        else:
            print("ERROR: Must provide metrics")
            exit(1)

        transforms = Compose([Resolution(1024, 1024), Bands(61, 61, None)])

    results = run_dwt(
        rgb_in_files,
        msi_in_files,
        msi_out_files,
        method,
        wavelet,
        level,
        metrics,
        dir,
        transforms,
    )

    print("Results calculated:")
    if "ssim" in results.keys():
        print(f"SSIM: {results['ssim'][0]}")
    if "sam" in results.keys():
        print(f"sam: {results['sam']}")
    if "psnr" in results.keys():
        print(f"psnr: {results['psnr']}")
    print("------------------")
