import torch
import torch.nn as nn
import torch.nn.functional as F

EPS = 1e-6


class PrototypeBank(nn.Module):
    """Global foreground / background prototypes (feat_dim vectors).

    Aggregated on the server by a count-weighted mean of client prototypes, then
    broadcast back. `initialized` guards the warm-up period where no consensus
    exists yet.
    """

    def __init__(self, feat_dim, tau=0.1):
        super().__init__()
        self.tau = tau
        self.register_buffer('proto_fg', torch.zeros(feat_dim))
        self.register_buffer('proto_bg', torch.zeros(feat_dim))
        self.register_buffer('initialized', torch.zeros(1))

    def set_global(self, proto_fg, proto_bg):
        self.proto_fg.copy_(proto_fg)
        self.proto_bg.copy_(proto_bg)
        self.initialized.fill_(1.0)

    def is_ready(self):
        return float(self.initialized) > 0.5

    def segment(self, feat):
        """Non-parametric segmentation from features.

        feat: (B, C, H, W). Returns p_proto (B,1,H,W) in [0,1], the probability
        a pixel is closer (cosine) to the fg prototype than the bg prototype.
        """
        fhat = F.normalize(feat, dim=1)
        pf = F.normalize(self.proto_fg, dim=0)
        pb = F.normalize(self.proto_bg, dim=0)
        sim_fg = (fhat * pf.view(1, -1, 1, 1)).sum(1, keepdim=True)
        sim_bg = (fhat * pb.view(1, -1, 1, 1)).sum(1, keepdim=True)
        return torch.sigmoid((sim_fg - sim_bg) / self.tau)


@torch.no_grad()
def local_prototypes(feat, mask, weight=None):
    """Count-weighted fg/bg mean feature vectors on one client.

    feat  : (B, C, H, W)
    mask  : (B, 1, H, W) in [0,1] (the refined target, hard-thresholded outside)
    weight: (B, 1, H, W) optional per-pixel confidence weight.
    Returns (fg_vec[C], fg_count, bg_vec[C], bg_count) or None if degenerate.
    """
    B, C, H, W = feat.shape
    if weight is None:
        weight = torch.ones_like(mask)
    fmask = (mask > 0.5).float() * weight
    bmask = (mask <= 0.5).float() * weight
    f = feat.permute(0, 2, 3, 1).reshape(-1, C)
    fm = fmask.reshape(-1)
    bm = bmask.reshape(-1)
    fg_cnt = fm.sum()
    bg_cnt = bm.sum()
    if fg_cnt < 1 or bg_cnt < 1:
        return None
    fg_vec = (f * fm.unsqueeze(1)).sum(0) / (fg_cnt + EPS)
    bg_vec = (f * bm.unsqueeze(1)).sum(0) / (bg_cnt + EPS)
    return fg_vec.detach(), float(fg_cnt), bg_vec.detach(), float(bg_cnt)


@torch.no_grad()
def refine_target(y_cam, label, p_teacher, p_proto, alpha, use_model, weight_floor=0.1):
    """Reliability-gated fusion of the CAM prior and the model consensus.

    y_cam    : (B,1,H,W) in {0,1}  -- the loaded CAM pseudo-mask (noisy prior)
    label    : (B,)                -- image-level binary label (hard guard)
    p_teacher: (B,1,H,W) in [0,1]  -- EMA teacher probability
    p_proto  : (B,1,H,W) in [0,1]  -- prototype-consensus probability
    alpha    : scalar in [0,1]     -- how much to trust the model over the CAM
    use_model: bool                -- False during warm-up (train on CAM only)

    Returns (soft_target, weight), both (B,1,H,W), detached.
    """
    if use_model:
        model_est = 0.5 * p_teacher + 0.5 * p_proto
        soft = (1.0 - alpha) * y_cam + alpha * model_est
        # Two INDEPENDENT model views (teacher params vs. prototype geometry).
        # Where they agree the target is trustworthy; where they disagree the
        # CAM is likely wrong AND the model is unsure -> down-weight.
        agree = 1.0 - (p_teacher - p_proto).abs()
        weight = agree.clamp(min=weight_floor)
    else:
        soft = y_cam.clone()
        weight = torch.ones_like(y_cam)

    # Hard guard from the image-level label: no tumor -> empty target, full
    # confidence. This is the one piece of supervision that is never noisy.
    lab = label.view(-1, 1, 1, 1).float().to(soft.device)
    pos = (lab > 0.5).float()
    soft = soft * pos
    weight = weight * pos + (1.0 - pos)  # weight=1 everywhere on negative slices
    return soft.detach(), weight.detach()


def weighted_dice_bce(logits, soft_target, weight, bce_w=1.0, dice_w=1.0, smooth=1.0):
    """DiceBCE with a per-pixel weight map."""
    bce = F.binary_cross_entropy_with_logits(logits, soft_target, reduction='none')
    bce = (bce * weight).sum() / (weight.sum() + EPS)

    p = torch.sigmoid(logits)
    w = weight
    inter = (w * p * soft_target).flatten(1).sum(1)
    denom = (w * p).flatten(1).sum(1) + (w * soft_target).flatten(1).sum(1)
    dice = (2 * inter + smooth) / (denom + smooth)
    dice_loss = 1.0 - dice.mean()
    return bce_w * bce + dice_w * dice_loss


def prototype_align_loss(feat, soft_target, weight, bank, conf_thr=0.6):
    """Pull confident fg/bg pixels toward the shared global prototypes.

    Only confident pixels (weight >= conf_thr) contribute, so noisy pixels do
    not corrupt the modality-invariant feature space. Returns a scalar tensor.
    """
    if not bank.is_ready():
        return feat.new_zeros(())
    fhat = F.normalize(feat, dim=1)                         # (B,C,H,W)
    pf = F.normalize(bank.proto_fg, dim=0).view(1, -1, 1, 1)
    pb = F.normalize(bank.proto_bg, dim=0).view(1, -1, 1, 1)
    cos_fg = (fhat * pf).sum(1, keepdim=True)               # (B,1,H,W)
    cos_bg = (fhat * pb).sum(1, keepdim=True)

    conf = (weight >= conf_thr).float()
    fg = (soft_target > 0.5).float() * conf
    bg = (soft_target <= 0.5).float() * conf
    # want fg pixels close to fg proto (and far from bg proto), vice versa.
    loss_fg = ((1.0 - cos_fg) + F.relu(cos_bg)) * fg
    loss_bg = ((1.0 - cos_bg) + F.relu(cos_fg)) * bg
    num = (fg.sum() + bg.sum() + EPS)
    return (loss_fg.sum() + loss_bg.sum()) / num


@torch.no_grad()
def dice_score(logits, target, threshold=0.5, eps=1e-6):
    """Per-sample whole-tumor Dice; empty-GT + empty-pred == 1.0."""
    pred = (torch.sigmoid(logits) > threshold).float().flatten(1)
    target = (target > 0.5).float().flatten(1)
    inter = (pred * target).sum(1)
    denom = pred.sum(1) + target.sum(1)
    dice = (2 * inter + eps) / (denom + eps)
    both_empty = (pred.sum(1) == 0) & (target.sum(1) == 0)
    dice = torch.where(both_empty, torch.ones_like(dice), dice)
    return dice.mean().item()
