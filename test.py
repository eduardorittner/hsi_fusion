from dataset import IcasspDataset
from networks.baseline import Baseline
from torch.utils.data import DataLoader
import torch
import torch.nn.functional as F
from utils import checkpoint, read_yaml
from transforms.get import get_transform

res = 128
bands = 31

train = IcasspDataset(
    "/home/erittner/docs/IC/icassp/data/processed/",
    transform=get_transform("crop(128, 512)|res(128, 256)|bands(4, 31)"),
    res=res,
    bands=bands,
)
test = IcasspDataset(
    "/home/erittner/docs/IC/icassp/data/test/",
    transform=get_transform("crop(128, 512)|res(128, 256)|bands(4, 31)"),
    res=res,
    bands=bands,
)

train_loader = DataLoader(train, batch_size=1, shuffle=True)
test_loader = DataLoader(test, batch_size=1, shuffle=True)

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)

cnn = Baseline().to(device)

EPOCHS = 50
optimizer = torch.optim.Adam(cnn.parameters())

# cnn.train()

data, target = next(iter(train_loader))
data = data.to(device=device).view(-1, bands, res, res)
target = target.to(device=device).view(bands, res, res)

for epoch in range(EPOCHS):
    print(f"Epochs [{epoch}/{EPOCHS}]")
    # for batch_idx, (data, target) in enumerate(train_loader):

    optimizer.zero_grad()

    pred = cnn(data).view(bands, res, res)
    print(pred.shape, target.shape)
    loss = F.cross_entropy(pred, target)
    print(loss.cpu().data.item())

    loss.backward()
    optimizer.step()
