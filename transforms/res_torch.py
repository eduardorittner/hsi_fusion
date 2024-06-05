import torch
from torch.nn.functional import interpolate
from typing import Optional


class ResTorch:
    def __init__(self, low_res: int, high_res: int):
        self.low_res = low_res
        self.high_res = high_res

    def __call__(self, x: torch.Tensor, y: torch.Tensor, z: Optional[torch.Tensor]):
        if z is not None:
            return interpolate(x), interpolate(y), interpolate(z)

        return interpolate(x), interpolate(y)
