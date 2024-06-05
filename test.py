from dataset import IcasspDataModule
from networks.baseline import Baseline
from torch.utils.data import DataLoader
import torch
import torch.nn.functional as F
from utils import checkpoint, read_yaml
from transforms.get import get_transform

hparams = read_yaml.read_yaml("/home/eduardo/hsi_fusion/models/example.yaml", False)

dataloader = IcasspDataModule(hparams)

dataloader.setup()

train_loader = dataloader.train_dataloader()
test_loader = dataloader.test_dataloader()
val_loader = dataloader.val_dataloader()

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)

print(train_loader, test_loader, val_loader)
