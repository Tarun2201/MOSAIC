import torch
import torch.nn.functional as F

# reuse v1 machinery
from losses_map import (  # noqa: F401
    PrototypeBank, prototype_align_loss, local_prototypes, dice_score,
)

EPS = 1e-6


@torch.no_grad()
def arbitrate_target(y_cam, label, p_teacher, p_proto, alpha, use_model,
                     beta=6.0, weight_floor=0.1):
    """Soft cross-source arbitration.

    vote = sigmoid(beta * [ (cam-.5) + (teacher-.5) + (proto-.5) ]) is a softened
    majority vote: any two confident sources decide the pixel. This lets the
    teacher + prototype recover foreground the CAM missed, and lets the models
    veto a CAM false positive.

    target = (1-alpha)*cam + alpha*vote   (alpha ramps 0 -> alpha_max)
    weight = decisiveness of the vote (|2*vote-1|), floored.

    Negative slices (image-level label 0) -> empty target, full weight.
    """
    if use_model:
        s = (y_cam - 0.5) + (p_teacher - 0.5) + (p_proto - 0.5)
        vote = torch.sigmoid(beta * s)
        soft = (1.0 - alpha) * y_cam + alpha * vote
        weight = (2.0 * vote - 1.0).abs().clamp(min=weight_floor)
    else:
        soft = y_cam.clone()
        weight = torch.ones_like(y_cam)

    lab = label.view(-1, 1, 1, 1).float().to(soft.device)
    pos = (lab > 0.5).float()
    soft = soft * pos
    weight = weight * pos + (1.0 - pos)
    return soft.detach(), weight.detach()


def focal_tversky(logits, target, weight, alpha=0.3, beta=0.7, gamma=1.0,
                  smooth=1.0, bce_w=0.5):
    """Weighted Focal-Tversky (+ a little weighted BCE for gradient stability).

    alpha weights false positives, beta weights false negatives. beta>alpha ->
    recall-favouring. gamma>1 focuses on hard (low-Tversky) samples.
    """
    p = torch.sigmoid(logits)
    w = weight
    tp = (w * p * target).flatten(1).sum(1)
    fp = (w * p * (1 - target)).flatten(1).sum(1)
    fn = (w * (1 - p) * target).flatten(1).sum(1)
    tversky = (tp + smooth) / (tp + alpha * fp + beta * fn + smooth)
    ft = (1.0 - tversky).clamp(min=EPS) ** gamma

    bce = F.binary_cross_entropy_with_logits(logits, target, reduction='none')
    bce = (bce * w).sum() / (w.sum() + EPS)
    return ft.mean() + bce_w * bce


def gated_crf_loss(prob, image, kernel=5, sigma_int=0.15, sigma_xy=6.0):
    """Intensity-gated pairwise affinity (gated-CRF style, on the raw MRI).

    Penalises label disagreement between nearby pixels in proportion to how
    similar their intensities are:  L = mean_i mean_j  w_ij * |p_i - p_j|,
    w_ij = exp(-|x_i-x_j|^2/2s_xy^2) * exp(-|I_i-I_j|^2/2s_int^2).
    Minimising it smooths within intensity-homogeneous regions and keeps edges,
    so the mask grows to fill the tumor and snaps to its boundary.

    prob : (B,1,H,W) in [0,1]        image: (B,Cin,H,W) input modalities
    """
    B, _, H, W = prob.shape
    intensity = image.mean(1, keepdim=True)  # (B,1,H,W) anatomical intensity
    pad = kernel // 2

    prob_u = F.unfold(prob, kernel, padding=pad).view(B, kernel * kernel, H, W)
    int_u = F.unfold(intensity, kernel, padding=pad).view(B, kernel * kernel, H, W)

    # spatial term (fixed per offset)
    ar = torch.arange(kernel, device=prob.device, dtype=prob.dtype) - pad
    dy, dx = torch.meshgrid(ar, ar, indexing='ij')
    d2 = (dy ** 2 + dx ** 2).reshape(-1)  # (kk,)
    w_xy = torch.exp(-d2 / (2 * sigma_xy ** 2)).view(1, -1, 1, 1)

    # intensity term
    w_int = torch.exp(-((int_u - intensity) ** 2) / (2 * sigma_int ** 2))
    w = w_xy * w_int                      # (B,kk,H,W)
    diff = (prob_u - prob).abs()          # broadcast center prob (B,1,H,W)
    loss_map = (w * diff).sum(1) / (w.sum(1) + EPS)
    return loss_map.mean()
