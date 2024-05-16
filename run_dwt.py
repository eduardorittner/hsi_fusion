from dwt.dwt import fuse_3dDWT
import argparse
from os.path import join, isdir
from tqdm import trange
import glob
from typing import Dict, List, Callable
from transforms.resolution import Resolution
from transforms.bands import Bands
from transforms.compose import Compose
from metrics.ssim import metric_ssim, metric_join_ssim
from metrics.sam import metric_sam, metric_join_sam
from metrics.psnr import metric_psnr, metric_join_psnr
import numpy as np
from utils.image_id import image_id
from datetime import datetime


def run_dwt(
    rgb_in: List[str],
    msi_in: List[str],
    msi_out: List[str],
    wavelet: type[List[str] | str],
    level: int,
    metrics: List[str],
    dir: str,
    transforms: Callable,
):
    results = {}

    results["method"] = method
    results["wavelet"] = wavelet
    results["level"] = level
    results["dir"] = dir

    for metric in metrics:
        results[metric] = {}

    print(
        f"""
Reading files from: {args.source}
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

        if "ssim" in metrics:
            results["ssim"][image_id(rgb_in_files[i])] = metric_ssim(result, expected)
        if "sam" in metrics:
            results["sam"][image_id(rgb_in_files[i])] = metric_sam(result, expected)
        if "psnr" in metrics:
            results["psnr"][image_id(rgb_in_files[i])] = metric_psnr(result, expected)

    if "ssim" in metrics:
        results["ssim"] = metric_join_ssim(results["ssim"])
    if "sam" in metrics:
        results["sam"] = metric_join_sam(results["sam"])
    if "psnr" in metrics:
        results["psnr"] = metric_join_psnr(results["psnr"])

    save_results(results, dir)

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="fusion-runner",
        description="Runs the given fusion method and calculates the given metrics",
    )

    parser.add_argument("source", type=str, help="Source directory")
    parser.add_argument("-m", "--method", type=str, help="Fusion method")
    parser.add_argument("-w", "--wavelet", type=str, help="Wavelet")
    parser.add_argument("-l", "--level", type=int, help="dwt level")
    parser.add_argument("-d", "--dir", type=str, help="Where results are stored")
    parser.add_argument("-e", "--metrics", type=str, help="Which metrics to compute")

    args = parser.parse_args()

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
