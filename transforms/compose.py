from typing import List, Any, Tuple, Optional
import numpy as np


class Compose:
    def __init__(self, transform_list: List[Any]):
        self.transform_list = transform_list

    def __call__(
        self, x: np.ndarray, y: np.ndarray, z: Optional[np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        for transform in self.transform_list:
            x, y, z = transform(x, y, z)

        return x, y, z

    def __str__(self) -> str:
        buffer = ""
        for i, transform in enumerate(self.transform_list):
            buffer += f"{i}: {str(transform)}\n"

        return buffer
