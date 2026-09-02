import torch
from .model import TinyUNet
def predict_tensor(x, num_classes=4):
    model=TinyUNet(num_classes=num_classes).eval()
    with torch.no_grad(): return model(x).argmax(1)
