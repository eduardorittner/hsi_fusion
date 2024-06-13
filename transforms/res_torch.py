import torch
from torch.nn.functional import interpolate
from typing import Optional


class ResTorch:
    def __init__(self, low_res: int, high_res: int, final_res: int):
        self.low_res = low_res
        self.high_res = high_res
        self.final_res = final_res

    def __call__(
        self,
        msi_in: torch.Tensor,
        hsi_in: torch.Tensor,
        hsi_out: Optional[torch.Tensor],
    ):

        if hsi_out is not None:
            return (
                interpolate(
                    msi_in, size=(self.final_res, self.final_res, msi_in.size()[2])
                ),
                interpolate(
                    hsi_in, size=(self.final_res, self.final_res, hsi_in.size()[2])
                ),
                interpolate(
                    hsi_out, size=(self.final_res, self.final_res, hsi_out.size()[2])
                ),
            )

        return interpolate(
            msi_in, size=(self.final_res, self.final_res, msi_in.size()[2])
        ), interpolate(hsi_in, size=(self.final_res, self.final_res, hsi_in.size()[2]))
