"""Plain 2D U-Net baseline (4 input MRI modalities -> 4 classes)."""
import torch
import torch.nn as nn


def block(i, o):
    return nn.Sequential(
        nn.Conv2d(i, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True),
        nn.Conv2d(o, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True),
    )


class UNet2D(nn.Module):
    def __init__(self, in_ch=4, n_classes=4, base=32):
        super().__init__()
        c = [base * 2 ** i for i in range(5)]  # 32,64,128,256,512
        self.enc = nn.ModuleList([block(in_ch, c[0])] + [block(c[i], c[i + 1]) for i in range(4)])
        self.pool = nn.MaxPool2d(2)
        self.up = nn.ModuleList([nn.ConvTranspose2d(c[i + 1], c[i], 2, stride=2) for i in range(4)])
        self.dec = nn.ModuleList([block(c[i] * 2, c[i]) for i in range(4)])
        self.head = nn.Conv2d(c[0], n_classes, 1)

    def forward(self, x):
        skips = []
        for i, e in enumerate(self.enc):
            x = e(x)
            if i < 4:
                skips.append(x)
                x = self.pool(x)
        for i in reversed(range(4)):
            x = self.up[i](x)
            x = self.dec[i](torch.cat([x, skips[i]], dim=1))
        return self.head(x)
