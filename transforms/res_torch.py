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

        msi_in = torch.unsqueeze(msi_in, 0)
        hsi_in = torch.unsqueeze(hsi_in, 0)
        msi_in = torch.unsqueeze(msi_in, 1)
        hsi_in = torch.unsqueeze(hsi_in, 1)

        if hsi_out is not None:
            hsi_out = torch.unsqueeze(hsi_out, 0)
            hsi_out = torch.unsqueeze(hsi_out, 1)

            return (
                torch.squeeze(
                    interpolate(
                        msi_in, size=(self.final_res, self.final_res, msi_in.size()[4])
                    )
                ),
                torch.squeeze(
                    interpolate(
                        hsi_in, size=(self.final_res, self.final_res, hsi_in.size()[4])
                    )
                ),
                torch.squeeze(
                    interpolate(
                        hsi_out,
                        size=(self.final_res, self.final_res, hsi_out.size()[4]),
                    )
                ),
            )

        return interpolate(
            msi_in, size=(self.final_res, self.final_res, msi_in.size()[2])
        ), interpolate(hsi_in, size=(self.final_res, self.final_res, hsi_in.size()[2]))
