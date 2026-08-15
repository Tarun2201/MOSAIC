import torch
import torch.nn as nn

from models_map import DoubleConv, ModalityAdapter  # reuse v1 blocks


class UNetFeatCls(nn.Module):
    """4-level U-Net. forward -> (seg_logits, feat_d1, cls_logit).

    feat_d1  : (B, base_filters, H, W)  last decoder feature (for prototypes)
    cls_logit: (B, 1)                   slice-level tumor-presence logit
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
        # presence head on the bottleneck (deepest, most semantic features)
        self.cls_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(f * 16, f * 4), nn.ReLU(inplace=True),
            nn.Dropout(0.3), nn.Linear(f * 4, 1),
        )
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
        seg_logits = self.out_conv(d1)
        cls_logit = self.cls_head(b)
        return seg_logits, d1, cls_logit

    def load_v1_backbone(self, state_dict, verbose=True):
        """Load a v1 UNetFeat state dict (no cls_head) with strict=False."""
        missing, unexpected = self.load_state_dict(state_dict, strict=False)
        if verbose:
            miss = [m for m in missing if not m.startswith('cls_head')]
            print(f"    warm-start: {len(miss)} non-cls params missing, "
                  f"{len(unexpected)} unexpected (cls_head initialised fresh)")


class SegPipelineCls(nn.Module):
    """adapter -> UNetFeatCls. Returns (seg_logits, feat, cls_logit)."""

    def __init__(self, adapter, unet):
        super().__init__()
        self.adapter = adapter
        self.unet = unet

    def forward(self, img):
        three = self.adapter(img)
        return self.unet(three)


__all__ = ["UNetFeatCls", "SegPipelineCls", "ModalityAdapter"]
