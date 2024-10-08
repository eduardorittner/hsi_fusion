import argparse
import numpy as np
import os
from metrics.sam import metric_sam
from metrics.psnr import metric_psnr
from metrics.ssim import metric_ssim


def name(file):
    return file.split("/")[-1].split(".")[0].split("pred")[-1].split("target")[-1]


def dostuff(a, b):
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("directory")
    args = parser.parse_args()

    tree = [x for x in os.walk(args.directory)]
    assert len(tree) == 1, "Data directory should be flat"
    dir, dirs, files = tree[0]
    record = {}
    for f in files:
        if f.endswith(".npy"):
            id = name(f)
            img = np.load(os.path.join(dir, f))
            prev = record.get(id)
            if prev is not None:
                pred, target = prev, img
                if "target" not in f:
                    pred, target = img, prev

                record[id] = (
                    metric_ssim(pred, target),
                    metric_sam(pred, target),
                    metric_psnr(target, pred),
                )
            else:
                record[id] = img

    ssim, sam, psnr = 0, 0, 0
    for value in record.values():
        ssim += value[0]
        sam += value[2]
        psnr += value[2]

    ssim /= len(record.values())
    sam /= len(record.values())
    psnr /= len(record.values())

    print(f"SSIM: {ssim}, SAM: {sam}, PSNR: {psnr}")
