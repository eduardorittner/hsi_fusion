from dwt.dwt import fuse_3dDWT, fuse_2dDWT
from dwt.average import fuse_average
import argparse
from os.path import join, isdir, isfile
from os import mkdir
from tqdm import tqdm
import glob
from typing import Dict, List, Callable
from transforms.get import get_transform
from transforms.resolution import Resolution
from transforms.bands import Bands
from transforms.compose import Compose
from metrics.ssim import metric_ssim
from metrics.sam import metric_sam
from metrics.psnr import metric_psnr
import numpy as np
from utils.image_id import image_id, mask_id
from utils.read_yaml import read_yaml
from datetime import datetime


def run_dwt(
    msi_in_files: List[str],
    hsi_in_files: List[str],
    hsi_out_files: List[str],
    mask_files: List[str] | None,
    method: str,
    wavelet: List[str] | str,
    level: int,
    metrics: List[str],
    use_mask: bool | None,
    dir: str,
    transforms: Callable | None,
) -> Dict | None:

    dir_str = ""

    if method == "3d-dwt":
        if isinstance(wavelet, str):
            dir_str = wavelet
        else:
            for wav in wavelet:
                dir_str += wav
                dir_str += "-"
            dir_str = dir_str[:-1]

    elif method == "average":
        dir_str = "average"

    elif method == "2d-dwt":
        dir_str = "2d"
        if isinstance(wavelet, str):
            dir_str += wavelet
        else:
            for wav in wavelet:
                dir_str += wav
                dir_str += "-"
            dir_str = dir_str[:-1]

    elif method == "baseline-msi":
        dir_str = "baseline-msi"

    elif method == "baseline-hsi":
        dir_str = "baseline-hsi"

    dir = dir + dir_str
    if use_mask:
        dir += "-mask"

    if isdir(dir):
        n_files = len(glob.glob(join(dir, "*.npy")))
        if n_files == len(msi_in_files):
            print("Results for this config have already been calculated.")
            return None
        print(f"{n_files} have already been calculated, resuming")

    else:
        n_files = 0
        mkdir(dir)

    with open(join(dir, "method.txt"), "w") as f:
        f.write(
            f"""Reading files from: {msi_in_files[0].split("/")[-3]}
{method} with wavelet(s): {wavelet}, level: {level}
metric(s): {metrics} stored in {dir}
"""
        )

    print(
        f"""Reading files from: {msi_in_files[0].split("/")[-3]}
{method} with wavelet(s): {wavelet}, level: {level}
metric(s): {metrics} stored in {dir}
    """
    )

    for i in tqdm(range(n_files, len(msi_in_files))):
        msi_in = np.load(msi_in_files[i])
        hsi_in = np.load(hsi_in_files[i])
        expected = np.load(hsi_out_files[i])

        if transforms is not None:
            msi_in, hsi_in, expected = transforms(msi_in, hsi_in, expected)

        if method == "3d-dwt":
            result = fuse_3dDWT(msi_in, hsi_in, wavelet, level, None)

        elif method == "2d-dwt":
            result = fuse_2dDWT(msi_in, hsi_in, wavelet, level, None)

        elif method == "average":
            result = fuse_average(msi_in, hsi_in, None)

        elif method == "baseline-msi":
            result = msi_in

        elif method == "baseline-msi":
            result = hsi_in

        else:
            print(f"[ERROR]: the method {method} not implemented")
            exit(1)

        id = image_id(msi_in_files[i])

        results = {}

        if use_mask:
            if mask_files is not None:
                mask = np.load(mask_files[i])

            else:
                print(
                    f"[ERROR]: mask flag is set to True but no mask files were provided"
                )
                exit(1)

            if mask_id(mask_files[i]) != id:
                print(f"[ERROR]: Mask id ({mask_id(mask_files[i])}) and image id ({id}) must be the same.")
                exit(1)

            mask = np.expand_dims(mask, np.argmin(result.shape))

            result = result * mask
            expected = expected * mask

        if "ssim" in metrics:
            results["ssim"] = metric_ssim(result, expected)
            print(results["ssim"])
        if "sam" in metrics:
            results["sam"] = metric_sam(result, expected)
            print(results["sam"])
        if "psnr" in metrics:
            results["psnr"] = metric_psnr(expected, result)
            print(results["psnr"])

        save_image_result(id, results, dir)


def save_image_result(image_id: str, results: Dict, dir: str):
    image_path = join(dir, image_id)
    np.save(image_path, np.array(results))


def access_metrics(r):
    # This is necessary to access a np array of type object

    return r["ssim"], r["sam"], r["psnr"]


def calculate_mean(dir: str, metrics: List[str]) -> Dict:
    files = sorted(glob.glob(dir + "*.npy"))

    results = {}
    for metric in metrics:
        results[metric] = 0

    for file in files:
        ssim, sam, psnr = np.vectorize(access_metrics)(np.load(file, allow_pickle=True))
        r = {"ssim": ssim, "sam": sam, "psnr": psnr}
        for metric in metrics:
            results[metric] += r[metric]

    for metric in metrics:
        results[metric] /= len(files)

    return results


def calculate_deviation(dir: str, metrics: List[str], results: Dict) -> Dict:
    files = sorted(glob.glob(dir + "*.npy"))

    deviation = None

    for file in files:
        ssim, sam, psnr = np.vectorize(access_metrics)(np.load(file, allow_pickle=True))
        r = {"ssim": ssim, "sam": sam, "psnr": psnr}
        if deviation is None:
            deviation = {}
            for metric in metrics:
                deviation[metric] = np.array((r[metric].shape))
                deviation[metric] = (r[metric] - results[metric]) ** 2
        else:
            for metric in metrics:
                deviation[metric] += (r[metric] - results[metric]) ** 2

    for metric in metrics:
        deviation[metric] /= len(files)
        deviation[metric] = np.sqrt(deviation[metric])

    return deviation


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
        if config.get("run") is None or config["run"] != "true":
            print(f"{file} is not set to run. Moving on to next one")
            continue

        print(f"Running {file}")
        msi_in_files = sorted(glob.glob(join(config["msi_in_files"], "*.npy")))
        hsi_in_files = sorted(glob.glob(join(config["hsi_in_files"], "*.npy")))
        hsi_out_files = sorted(glob.glob(join(config["hsi_out_files"], "*.npy")))

        mask_files = None

        if config.get("mask"):
            mask_files = sorted(glob.glob(join(config["mask_files"], "*.npy")))

        run_dwt(
            msi_in_files,
            hsi_in_files,
            hsi_out_files,
            mask_files,
            config["method"],
            config["wavelet"].split(","),
            config["level"],
            config["metrics"].split(","),
            config["mask"],
            config["dir"],
            get_transform(config["transforms"]),
        )
        print("Results calculated:")


def save_results_to_file(mean: Dict, deviation: Dict, dir: str):
    with open(join(dir, "results.txt"), "w") as f:
        for metric in mean.keys():
            f.write(f"Metric: {metric}\n")
            if mean[metric].shape:
                for a, b in zip(mean[metric], deviation[metric]):
                    f.write(f"{a} +- {b}\n")
            else:
                f.write(f"{mean[metric]} +- {deviation[metric]}\n")

            f.write("\n")


def aggregate_results(dir: str):
    folders = sorted(glob.glob(join(dir, "*")))
    results = {}

    for folder in folders:
        wav = folder.split("/")[-1]
        filename = join(folder, "results.txt")
        if isfile(filename):
            with open(filename, "r") as f:
                file = f.readlines()
                results[wav] = {"ssim": file[1], "sam": file[65], "psnr": file[68]}

    with open(join(dir, "results.txt"), "w") as f:
        for key, value in results.items():
            f.write(f"{key}\n")
            f.write(f"{value}\n")

    print(f"Saving results in {join(dir, 'results.txt')}")


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
    parser.add_argument("-r", "--results", type=str, help="Calculate results")
    parser.add_argument("-a", "--aggregate", type=str, help="Aggregate results")
    parser.add_argument(
        "-f", "--mask", type=bool, help="Whether to use metrics on mask or whole image"
    )

    args = parser.parse_args()

    if args.aggregate is not None:
        dir = args.aggregate
        aggregate_results(dir)
        exit(0)

    if args.results is not None:
        mean = calculate_mean(args.results, args.metrics.split(","))
        deviation = calculate_deviation(args.results, args.metrics.split(","), mean)
        mean_path = join(args.results, "mean")
        deviation_path = join(args.results, "deviation")
        save_results_to_file(mean, deviation, args.results)
        print(f"results saved in {args.results}")
        exit(0)

    if args.suite is not None:
        run_dwt_suite(args.suite)
        exit(0)

    if args.config is not None:
        # Read from config file
        config = read_yaml(args.config, False)
        msi_in_files = sorted(glob.glob(join(config["msi_in_files"], "*.npy")))
        hsi_in_files = sorted(glob.glob(join(config["hsi_in_files"], "*.npy")))
        hsi_out_files = sorted(glob.glob(join(config["hsi_out_files"], "*.npy")))

        mask_files = None

        if config.get("mask"):
            mask_files = sorted(glob.glob(join(config["mask_files"], "*.npy")))

        results = run_dwt(
            msi_in_files,
            hsi_in_files,
            hsi_out_files,
            mask_files,
            config["method"],
            config["wavelet"].split(","),
            config["level"],
            config["metrics"].split(","),
            config["mask"],
            config["dir"],
            get_transform(config["transforms"]),
        )

    else:
        if not isdir(args.source):
            print(f"ERROR: Directory {args.source} does not exist.")
            exit(1)

        msi_in_files = sorted(glob.glob(join(args.source, "msi_in/*.npy")))
        hsi_in_files = sorted(glob.glob(join(args.source, "hsi_in/*.npy")))
        hsi_out_files = sorted(glob.glob(join(args.source, "hsi_out/*.npy")))

        mask_files = None
        mask = args.mask

        if mask:
            mask_files = sorted(glob.glob(join(args.source, "masks/*.npy")))

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
            msi_in_files,
            hsi_in_files,
            hsi_out_files,
            mask_files,
            method,
            wavelet,
            level,
            metrics,
            mask,
            dir,
            transforms,
        )

        if results is not None:
            print("Results calculated:")
            if "ssim" in results.keys():
                print(f"SSIM: {results['ssim'][0]}")
            if "sam" in results.keys():
                print(f"sam: {results['sam']}")
            if "psnr" in results.keys():
                print(f"psnr: {results['psnr']}")
            print("------------------")
