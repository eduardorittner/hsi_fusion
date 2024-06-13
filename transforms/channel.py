import torch
from typing import Optional, Tuple


class Channel:
    def __init__(self, channel_first: bool):
        self.channel_first = channel_first

    def is_channel_first(self, x: torch.Tensor) -> bool:
        shape = x.size()
        return shape[0] < shape[1]

    def __call__(
        self, msi_in: torch.Tensor, hsi_in: torch.Tensor, out: Optional[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:

        if self.is_channel_first(msi_in) and not self.channel_first:
            msi_in = torch.movedim(msi_in, 0, 2)
        elif not self.is_channel_first(msi_in) and self.channel_first:
            msi_in = torch.movedim(msi_in, 2, 0)

        if self.is_channel_first(hsi_in) and not self.channel_first:
            hsi_in = torch.movedim(hsi_in, 0, 2)
        elif not self.is_channel_first(hsi_in) and self.channel_first:
            hsi_in = torch.movedim(hsi_in, 2, 0)

        if out is not None:
            if self.is_channel_first(out) and not self.channel_first:
                out = torch.movedim(out, 0, 2)
            elif not self.is_channel_first(out) and self.channel_first:
                out = torch.movedim(out, 2, 0)
            return msi_in, hsi_in, out

        return msi_in, hsi_in, None
