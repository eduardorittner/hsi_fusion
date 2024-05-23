import pywt
import numpy as np
from typing import List, Callable


class DWT2D_coeffs(object):
    def __init__(self, image: np.ndarray, wavelet: type[List[str] | str], level: int):
        self.wavelet = wavelet
        self.level = level
        self.coeffs, self.slices, self.shapes = pywt.ravel_coeffs(
            pywt.wavedec2(image, wavelet=wavelet, level=level)
        )
        self.coeffs = np.array(self.coeffs)

    def copy_coeffs(self):
        return np.copy(self.coeffs)

    def coeffs_approx(self):
        return (self.coeffs[self.slices[0]], self.slices[0])

    def coeffs_detail(self):
        results = []

        for i in range(self.level):
            results.append(
                (self.coeffs[self.slices[i + 1]["dd"]], self.slices[i + 1]["dd"])
            )
            results.append(
                (self.coeffs[self.slices[i + 1]["ad"]], self.slices[i + 1]["ad"])
            )
            results.append(
                (self.coeffs[self.slices[i + 1]["da"]], self.slices[i + 1]["da"])
            )

        return results


def fuse_approx_2d(rgb_in: DWT2D_coeffs, msi_in: DWT2D_coeffs, fused: np.ndarray):
    rgb = rgb_in.coeffs_approx()
    msi = msi_in.coeffs_approx()

    start, stop = rgb[1].start, rgb[1].stop

    fused[start:stop] = (rgb[0][start:stop] + msi[0][start:stop]) / 2


def fuse_detail_2d(rgb_in: DWT2D_coeffs, msi_in: DWT2D_coeffs, fused: np.ndarray):
    rgb = rgb_in.coeffs_detail()
    start, stop = rgb[1].start, rgb[1].stop
    fused[start:stop] = rgb[start:stop]


class DWT_coeffs(object):
    def __init__(self, image: np.ndarray, wavelet: type[List[str] | str], level: int):
        self.wavelet = wavelet
        self.level = level
        self.coeffs, self.slices, self.shapes = pywt.ravel_coeffs(
            pywt.wavedecn(image, wavelet=wavelet, level=level)
        )
        self.coeffs = np.array(self.coeffs)

    def copy_coeffs(self):
        return np.copy(self.coeffs)

    def coeffs_approx(self):
        return (self.coeffs[self.slices[0]], self.slices[0])

    def coeffs_spectral_detail(self):
        results = []

        for i in range(self.level):
            results.append(
                (self.coeffs[self.slices[i + 1]["aad"]], self.slices[i + 1]["aad"])
            )

        return results

    def coeffs_spatial_detail(self):
        results = []

        for i in range(self.level):
            results.append(
                (self.coeffs[self.slices[i + 1]["ada"]], self.slices[i + 1]["ada"])
            )
            results.append(
                (self.coeffs[self.slices[i + 1]["daa"]], self.slices[i + 1]["daa"])
            )
            results.append(
                (self.coeffs[self.slices[i + 1]["dda"]], self.slices[i + 1]["dda"])
            )

        return results

    def coeffs_detail_both(self):
        results = []

        for i in range(self.level):
            results.append(
                (self.coeffs[self.slices[i + 1]["add"]], self.slices[i + 1]["add"])
            )
            results.append(
                (self.coeffs[self.slices[i + 1]["dad"]], self.slices[i + 1]["dad"])
            )
            results.append(
                (self.coeffs[self.slices[i + 1]["ddd"]], self.slices[i + 1]["ddd"])
            )

        return results


def fuse_approx(rgb_in: DWT_coeffs, msi_in: DWT_coeffs, fused: np.ndarray):
    # Gets the average between each pixel
    rgb = rgb_in.coeffs_approx()
    msi = msi_in.coeffs_approx()

    start, stop = rgb[1].start, rgb[1].stop

    fused[start:stop] = (rgb[0][start:stop] + msi[0][start:stop]) / 2


def fuse_spatial_detail(rgb_in: DWT_coeffs, msi_in: DWT_coeffs, fused: np.ndarray):
    rgb = rgb_in.coeffs_spatial_detail()

    for level in rgb:
        start, stop = level[1].start, level[1].stop

        fused[start:stop] = level[0][: stop - start]


def fuse_spectral_detail(rgb_in: DWT_coeffs, msi_in: DWT_coeffs, fused: np.ndarray):
    msi = msi_in.coeffs_spectral_detail()

    for level in msi:
        start, stop = level[1].start, level[1].stop

        fused[start:stop] = level[0][: stop - start]

    return


def fuse_detail(rgb_in: DWT_coeffs, msi_in: DWT_coeffs, fused: np.ndarray):
    rgb = rgb_in.coeffs_detail_both()
    msi = msi_in.coeffs_detail_both()

    for level in rgb:
        start, stop = level[1].start, level[1].stop
        fused[start:stop] = (level[0][: stop - start] + level[0][: stop - start]) / 2

    return


def fuse_2dDWT(
    rgb_in: np.ndarray,
    msi_in: np.ndarray,
    wavelet: type[str | List[str]],
    level: int,
    transform: Callable,
) -> np.ndarray:
    if transform is not None:
        rgb_in, msi_in, _ = transform(rgb_in, msi_in, None)

    results = np.array(rgb_in.shape)

    for band in range(rgb_in.shape[2]):
        rgb_coeffs = DWT2D_coeffs(rgb_in[:, :, band], wavelet, level)
        msi_coeffs = DWT2D_coeffs(msi_in[:, :, band], wavelet, level)

        fused_coeffs = rgb_coeffs.copy_coeffs()
        fuse_approx_2d(rgb_coeffs, msi_coeffs, fused_coeffs)
        fuse_detail_2d(rgb_coeffs, msi_coeffs, fused_coeffs)

        results[:, :, band] = pywt.waverec2(
            pywt.unravel_coeffs(fused_coeffs, rgb_coeffs.slices, rgb_coeffs.shapes),
            wavelet=wavelet,
        )

    return results


def fuse_3dDWT(
    rgb_in: np.ndarray,
    msi_in: np.ndarray,
    wavelet: type[str | List[str]],
    level: int,
    transform: Callable,
) -> np.ndarray:

    if transform is not None:
        rgb_in, msi_in, _ = transform(rgb_in, msi_in, None)

    rgb_coeffs = DWT_coeffs(rgb_in, wavelet, level)
    msi_coeffs = DWT_coeffs(msi_in, wavelet, level)

    fused_coeffs = rgb_coeffs.copy_coeffs()

    fuse_approx(rgb_coeffs, msi_coeffs, fused_coeffs)
    fuse_spatial_detail(rgb_coeffs, msi_coeffs, fused_coeffs)
    fuse_spectral_detail(rgb_coeffs, msi_coeffs, fused_coeffs)
    fuse_detail(rgb_coeffs, msi_coeffs, fused_coeffs)

    result = pywt.waverecn(
        pywt.unravel_coeffs(fused_coeffs, rgb_coeffs.slices, rgb_coeffs.shapes),
        wavelet=wavelet,
    )

    # Wavelet transform return one extra duplicate band in the end
    if level == 2:
        result = np.delete(result, -1, 2)

    return result
