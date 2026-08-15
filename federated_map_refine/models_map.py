
import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """(conv -> InstanceNorm -> ReLU) * 2 (matches the pipeline's norm choice)."""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.InstanceNorm2d(out_ch, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.InstanceNorm2d(out_ch, affine=True),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class ModalityAdapter(nn.Module):
    """Personalized 1-stage U-Net, (in_channels -> 3). Kept LOCAL per client.

    Same spirit as the SimpleUNet used elsewhere, re-implemented here so the new
    directory is self-contained and does not depend on the old module.
    """

    def __init__(self, in_channels, width=64):
        super().__init__()
        self.down = nn.Conv2d(in_channels, width, 3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.up = nn.ConvTranspose2d(width, width, 2, stride=2)
        self.out_conv = nn.Conv2d(width, 3, 3, padding=1)
        self.n1 = nn.InstanceNorm2d(width)
        self.n2 = nn.InstanceNorm2d(width)

    def forward(self, x):
        x = F.relu(self.n1(self.down(x)))
        x = self.pool(x)
        x = F.relu(self.n2(self.up(x)))
        return self.out_conv(x)


class UNetFeat(nn.Module):
    """Classic 4-level U-Net; forward returns (logits, feat).

    feat is the last decoder feature `d1` of shape (B, base_filters, H, W). It is
    L2-normalisable per pixel and is used for prototype-based segmentation and
    the prototype-alignment loss.
    """

    def __init__(self, in_channels=3, out_channels=1, base_filters=32):
        super().__init__()
        f = base_filters
        self.enc1 = DoubleConv(in_channels, f)
        self.enc2 = DoubleConv(f, f * 2)
        self.enc3 = DoubleConv(f * 2, f * 4)
        self.enc4 = DoubleConv(f * 4, f * 8)
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = DoubleConv(f * 8, f * 16)
        self.up4 = nn.ConvTranspose2d(f * 16, f * 8, 2, stride=2)
        self.dec4 = DoubleConv(f * 16, f * 8)
        self.up3 = nn.ConvTranspose2d(f * 8, f * 4, 2, stride=2)
        self.dec3 = DoubleConv(f * 8, f * 4)
        self.up2 = nn.ConvTranspose2d(f * 4, f * 2, 2, stride=2)
        self.dec2 = DoubleConv(f * 4, f * 2)
        self.up1 = nn.ConvTranspose2d(f * 2, f, 2, stride=2)
        self.dec1 = DoubleConv(f * 2, f)
        self.out_conv = nn.Conv2d(f, out_channels, 1)
        self.feat_dim = f

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b = self.bottleneck(self.pool(e4))
        d4 = self.dec4(torch.cat([self.up4(b), e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        logits = self.out_conv(d1)
        return logits, d1


class SegPipeline(nn.Module):
    """adapter -> shared UNetFeat. Returns (logits, feat)."""

    def __init__(self, adapter, unet):
        super().__init__()
        self.adapter = adapter
        self.unet = unet

    def forward(self, img):
        three = self.adapter(img)
        logits, feat = self.unet(three)
        return logits, feat
