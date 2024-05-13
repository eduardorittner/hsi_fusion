from dataset import IcasspDataset
from networks.baseline import Baseline
from torch.utils.data import DataLoader
import torch
import torch.nn.functional as F
from utils import checkpoint, read_yaml
from typing import Dict, Any, Tuple
import argparse
import yaml


def train(
    model: nn.Module,
    metrics: Dict[str, object],
    configs: Dict[str, Any],
    train_dataset: IcasspDataset,
    val_dataset: IcasspDataset,
    epochs: int,
    total_iteration: int,
    ep: int = 0,
    iter: int = 0,
):

    # Choose loss method

    # Train loop
    #   Data loaders

    return
