import os
import yaml
from typing import Dict, Any


def read_yaml(file: str, nn: bool) -> Dict[str, Any]:
    with open(file, "r") as yaml_file:
        config = yaml.load(yaml_file, Loader=yaml.FullLoader)

    if nn:
        fcn = os.path.basename(file).replace(".yaml", "")
        print(f"WARNING: Setting fixed_checkpoint_name as {fcn}")
        config["fixed_checkpoint_name"] = fcn

    return config
