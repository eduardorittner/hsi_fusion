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

for thing in train_loader:
    print(thing)
train_in, train_out = train_loader(0)
test_in, test_out = test_loader(0)
val_in, val_out = val_loader(0)

print(f"train: in [{train_in.size}] - out [{train_out.size}]")
print(f"test: in [{test_in.size}] - out [{test_out.size}]")
print(f"val: in [{val_in.size}] - out [{val_out.size}]")

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)
