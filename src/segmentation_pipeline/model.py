import torch
from torch import nn


class ConvBlock(nn.Module):
    def __init__(self, c_in, c_out):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(c_in, c_out, 3, padding=1),
            nn.BatchNorm2d(c_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(c_out, c_out, 3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class TinyUNet(nn.Module):
    def __init__(self, in_channels=3, num_classes=4):
        super().__init__()
        self.enc1 = ConvBlock(in_channels, 16)
        self.pool = nn.MaxPool2d(2)
        self.enc2 = ConvBlock(16, 32)
        self.up = nn.ConvTranspose2d(32, 16, 2, stride=2)
        self.dec = ConvBlock(32, 16)
        self.head = nn.Conv2d(16, num_classes, 1)

    def forward(self, x):
        x1 = self.enc1(x)
        x2 = self.enc2(self.pool(x1))
        x = self.up(x2)
        x = torch.cat([x, x1], dim=1)
        return self.head(self.dec(x))
