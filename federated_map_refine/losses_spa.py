import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from federated_modality_alignment_losses import SpectralPrototypeAlignmentLoss  # noqa: E402

EPS = 1e-8


def _norm_bands(energy):
    """Per-channel normalize band energies to a distribution (sum=1 over bands)."""
    return energy / (energy.sum(dim=-1, keepdim=True) + EPS)   # (C, bands)


class SpectralAlign(nn.Module):
    """Whole-map spectral prototype alignment (SPA) on normalized band energies."""

    def __init__(self, num_channels=3, num_freq_bands=8, momentum=0.9):
        super().__init__()
        self._spec = SpectralPrototypeAlignmentLoss(num_channels=num_channels,
                                                    num_freq_bands=num_freq_bands)
        self.momentum = momentum
        self.register_buffer('global_dist', torch.ones(num_channels, num_freq_bands) / num_freq_bands)
        self.register_buffer('initialized', torch.zeros(1))

    def _dist(self, feat):
        return _norm_bands(self._spec.compute_spectral_energy(feat))

    @torch.no_grad()
    def compute_local_statistics(self, feat):
        return {'dist': self._dist(feat).detach()}

    def update_global_statistics(self, stats):
        if 'dist' in stats:
            self.global_dist = self.momentum * self.global_dist + (1 - self.momentum) * stats['dist']
            self.initialized.fill_(1.0)

    def is_ready(self):
        return float(self.initialized) > 0.5

    def forward(self, feat):
        if not self.is_ready():
            return feat.new_zeros(())
        return F.mse_loss(self._dist(feat), self.global_dist)


class RegionConditionedSpectralAlign(nn.Module):
    """Foreground/background-separated spectral alignment (normalized band dists).

    Only the tumor (foreground) spectral distribution is forced to match the
    cross-client consensus; the background is aligned only to its own consensus,
    since background appearance legitimately differs across MRI modalities.
    """

    def __init__(self, num_channels=3, num_freq_bands=8, momentum=0.9, min_fg_pixels=50.0):
        super().__init__()
        self._spec = SpectralPrototypeAlignmentLoss(num_channels=num_channels,
                                                    num_freq_bands=num_freq_bands)
        self.momentum = momentum
        self.min_fg_pixels = min_fg_pixels
        self.register_buffer('global_fg', torch.ones(num_channels, num_freq_bands) / num_freq_bands)
        self.register_buffer('global_bg', torch.ones(num_channels, num_freq_bands) / num_freq_bands)
        self.register_buffer('initialized', torch.zeros(1))

    def _dists(self, feat, mask):
        fg = _norm_bands(self._spec.compute_spectral_energy(feat * mask))
        bg = _norm_bands(self._spec.compute_spectral_energy(feat * (1.0 - mask)))
        return fg, bg

    @torch.no_grad()
    def compute_local_statistics(self, feat, mask):
        fg, bg = self._dists(feat, mask)
        out = {'bg_freq': bg.detach()}
        if float(mask.sum()) > self.min_fg_pixels:
            out['fg_freq'] = fg.detach()
        return out

    def update_global_statistics(self, stats):
        if 'fg_freq' in stats:
            self.global_fg = self.momentum * self.global_fg + (1 - self.momentum) * stats['fg_freq']
        if 'bg_freq' in stats:
            self.global_bg = self.momentum * self.global_bg + (1 - self.momentum) * stats['bg_freq']
        self.initialized.fill_(1.0)

    def is_ready(self):
        return float(self.initialized) > 0.5

    def forward(self, feat, mask):
        if not self.is_ready():
            return feat.new_zeros(())
        fg, bg = self._dists(feat, mask)
        loss = F.mse_loss(bg, self.global_bg)
        if float(mask.sum()) > self.min_fg_pixels:
            loss = loss + F.mse_loss(fg, self.global_fg)
        return loss


def aggregate_freq_stats(client_stats, weights):
    """Weighted mean of per-client normalized-band-distribution dicts."""
    valid = [(s, w) for s, w in zip(client_stats, weights) if s]
    if not valid:
        return None
    keys = set()
    for s, _ in valid:
        keys.update(s.keys())
    agg = {}
    for k in keys:
        vw = [(s[k], w) for s, w in valid if k in s]
        if vw:
            wsum = sum(v * w for v, w in vw)
            tw = sum(w for _, w in vw)
            agg[k] = wsum / tw
    return agg
