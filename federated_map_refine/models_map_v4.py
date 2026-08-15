import torch
import torch.nn as nn

from models_map_v2 import ModalityAdapter, UNetFeatCls


class SegPipeline25D(nn.Module):
    """(K, num_mod) stack -> shared adapter per slice -> 3K ch -> UNetFeatCls.

    forward(img_stack): img_stack (B, K, num_mod, H, W) -> (seg_logits, feat, cls_logit)
    """

    def __init__(self, adapter, unet, K):
        super().__init__()
        self.adapter = adapter
        self.unet = unet
        self.K = K

    def forward(self, img_stack):
        B, K, C, H, W = img_stack.shape
        x = img_stack.reshape(B * K, C, H, W)
        three = self.adapter(x)                       # (B*K, 3, H, W)
        three = three.reshape(B, K * 3, H, W)         # centre block = [3*(K//2):+3]
        return self.unet(three)


def build_v4_unet(base_filters, K, device):
    return UNetFeatCls(in_channels=3 * K, out_channels=1, base_filters=base_filters).to(device)


@torch.no_grad()
def warm_start_25d(unet, v2_state, K, verbose=True):
    """Load a v2 UNetFeatCls state dict into the 2.5D backbone.

    All layers load directly except the first conv (enc1.net.0.weight), whose
    input channels grew 3 -> 3K. We place the v2 weights on the CENTRE-slice
    block and zero the neighbour blocks, so the model is identical to v2 at init.
    """
    first_key = 'enc1.net.0.weight'
    v2_first = v2_state.get(first_key)
    sd = {k: v for k, v in v2_state.items() if k != first_key}
    missing, unexpected = unet.load_state_dict(sd, strict=False)
    if v2_first is not None:
        w = torch.zeros_like(unet.enc1.net[0].weight)   # (f, 3K, 3, 3)
        c = (K // 2) * 3
        w[:, c:c + 3] = v2_first.to(w.device)
        unet.enc1.net[0].weight.data.copy_(w)
    if verbose:
        miss = [m for m in missing if not m.startswith('enc1.net.0')]
        print(f"    v4 warm-start: centre-block init from v2, {len(miss)} other missing")


__all__ = ["SegPipeline25D", "ModalityAdapter", "UNetFeatCls", "build_v4_unet", "warm_start_25d"]
