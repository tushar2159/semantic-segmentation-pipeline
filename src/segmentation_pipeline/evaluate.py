import torch

from .data import SyntheticSegmentationDataset
from .metrics import mean_iou
from .model import TinyUNet
from .utils import load_config


def evaluate(config_path="config/default.yaml"):
    cfg = load_config(config_path)
    ds = SyntheticSegmentationDataset(4, cfg["image_size"], cfg["num_classes"], seed=99)
    model = TinyUNet(num_classes=cfg["num_classes"]).eval()
    x, y = ds[0]
    with torch.no_grad():
        logits = model(x.unsqueeze(0))
    return {"mean_iou": mean_iou(logits, y.unsqueeze(0), cfg["num_classes"])}
