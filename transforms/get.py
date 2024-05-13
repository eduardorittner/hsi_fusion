from .crop import Crop
from .resolution import Resolution
from .bands import Bands
from .compose import Compose


def get_transform(transform: str):
    if transform == "256x31_1024x4":
        return Compose([Bands(4, 31, None), Resolution(256, 1024)])

    if transform == "512x31_512x4":
        return Compose([Bands(4, 31, None), Resolution(512, 512)])

    if transform == "crop(128, 512)|res(128, 256)|bands(4, 31)":
        return Compose(
            [
                Crop(512, 512 // 2, 128, 128 // 2),
                Resolution(128, 256),
                Bands(4, 31, None),
            ]
        )
