from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .data import SyntheticSegmentationDataset
from .model import TinyUNet
from .utils import load_config, resolve_device, set_seed


def train(config_path="config/default.yaml"):
    cfg = load_config(config_path)
    set_seed(cfg["seed"])
    device = resolve_device(cfg["device"])
    ds = SyntheticSegmentationDataset(16, cfg["image_size"], cfg["num_classes"], cfg["seed"])
    dl = DataLoader(ds, batch_size=cfg["batch_size"], shuffle=True)
    model = TinyUNet(num_classes=cfg["num_classes"]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["learning_rate"])
    loss_fn = torch.nn.CrossEntropyLoss()
    model.train()
    for _ in range(cfg["epochs"]):
        for x, y in dl:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            opt.step()
    out = Path(cfg["output_dir"])
    out.mkdir(parents=True, exist_ok=True)
    ckpt = out / "model.pt"
    torch.save(model.state_dict(), ckpt)
    return ckpt
