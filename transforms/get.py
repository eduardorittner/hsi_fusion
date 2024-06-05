from .crop import Crop
from .resolution import Resolution
from .res_torch import ResTorch
from .bands import Bands
from .compose import Compose


def get_transform(transform: str):
    if transform == "256x31_1024x4":
        return Compose([Bands(4, 31, None), Resolution(256, 1024)])

    if transform == "512x31_512x4":
        return Compose([Bands(4, 31, None), Resolution(512, 512)])

    if transform == "1024x61_1024x61":
        return Compose([Resolution(1024, 1024), Bands(61, 61, None)])

    if transform == "crop(128, 512)|res(128, 256)|bands(4, 31)":
        return Compose(
            [
                Crop(512, 512 // 2, 128, 128 // 2),
                Resolution(128, 256),
                Bands(4, 31, None),
            ]
        )

    if transform == "512x512":
        return ResTorch(256, 512, 512)

    return None
