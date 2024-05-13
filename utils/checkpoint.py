import torch
from torch import nn
import os
import glob
from typing import Any, Dict


def save_checkpoint(
    model_path: str,
    epoch: int,
    iteration: int,
    model: nn.Module,
    optimizer,
    best_loss,
    checkpoint_name,
    disable=False,
):
    state = {
        "epoch": epoch,
        "iter": iteration,
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "best_loss": best_loss,
    }

    if disable:
        print("WARNING: Checkpoint saving disabled")
        return
    torch.save(state, os.path.join(model_path, f"{checkpoint_name}.pth"))


def load_checkpoint(model: nn.Module, configs: Dict[str, Any], exact_file_name=None):

    ckpt_path = configs["save_checkpoint_path"]

    if exact_file_name is None:
        latest_ckpt = os.path.join(ckpt_path, f"{configs['fixed_checkpoint_name']}.pth")
        print(f"Looking for checkpoint in {latest_ckpt}")

        if not os.path.exists(latest_ckpt):
            print(f"{latest_ckpt} doesn't exist, looking in {ckpt_path}")

            if not os.path.exist(ckpt_path):
                raise ValueError(f"Checkpoint folder {ckpt_path} does not exist")

            list_of_ckpts = glob.glob(ckpt_path + "/best*")

            if len(list_of_ckpts) < 0:
                print(
                    "Warning: There are no checkpoints named 'best...', looking for latest checkpoints"
                )
                list_of_ckpts = glob.glob(ckpt_path + "*pth")

                if len(list_of_ckpts) < 0:
                    raise ValueError(f"No checkpoints found in {ckpt_path}")

            print(f"{len(list_of_ckpts)} checkpoints found, loading the latest one")

            latest_ckpt = max(list_of_ckpts, key=os.path.getctime)

    else:
        latest_ckpt = exact_file_name
        print(f"Loading from checkpoint {exact_file_name}")
        if not os.path.exists(latest_ckpt):
            raise ValueError(f"No file named {latest_ckpt}")

    if latest_ckpt is not None:
        if os.path.isfile(latest_ckpt):
            print(f"Loading model from checkpoint {latest_ckpt}")

            ckpt = torch.load(latest_ckpt, map_location=torch.device("cpu"))
            model.load_state_dict(ckpt["state_dict"])

            best_loss = ckpt.get("best_loss")
            if best_loss is None:
                print(
                    "Forcing best_loss = 1000 because no saved best_loss value was found"
                )
                best_loss = 1000

            print(f"Loaded checkpoint from epoch {ckpt['epoch']}")

            return model, latest_ckpt, best_loss

    return None, None, None
