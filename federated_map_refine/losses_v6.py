import math
import torch
import torch.nn.functional as F

EPS = 1e-6


def mil_positive_loss(logits, label, k=16):
    """Top-k multiple-instance positivity on POSITIVE slices only.

    On a slice with image-level label 1 we know >=1 tumor pixel exists, so the k
    highest logits must be foreground: BCE(topk_logits, 1). Self-targeting -- once
    the peak is confident the loss is ~0, so it acts only on slices the model
    (mis)predicts as empty. label==0 slices contribute nothing.

    logits: (B,1,H,W)   label: (B,) in {0,1}
    """
    B = logits.shape[0]
    lab = label.view(B).float().to(logits.device)
    m = lab > 0.5
    if m.sum() < 1:
        return logits.new_zeros(())
    flat = logits.view(B, -1)[m]                       # (n_pos, HW)
    kk = min(k, flat.shape[1])
    top = flat.topk(kk, dim=1).values                  # (n_pos, k)
    return F.binary_cross_entropy_with_logits(top, torch.ones_like(top))


def mil_rescue_loss(logits, label, cls_logit, p_teacher, rho_lo,
                    k=16, teacher_thr=0.3):
    """Precise recall rescue. GLOBAL forcing (a size floor applied to every
    positive slice) shifts the shared backbone's operating point, making the
    weakest client hallucinate and collapsing its specificity. This term instead
    fires ONLY on slices that are almost certainly genuine misses, and
    self-limits, so it recovers recall without moving the operating point:

      gate = label==1                     (image-level tumor present)
           & sigmoid(cls) > 0.5           (the CLEAN presence head agrees)
           & teacher_max_prob > teacher_thr (the EMA teacher sees latent evidence
                                            -> we grow real signal, not noise)
           & pred_fraction < rho_lo       (the model is UNDER-predicting; once it
                                            grows past the federated size floor the
                                            gate turns off -> no over-growth)

    On gated slices push the top-k logits to foreground, weighted by the teacher's
    confidence. Negative slices are never touched, so specificity is protected by
    construction. rho_lo comes from the federated size prior.

    logits, p_teacher: (B,1,H,W)   label: (B,)   cls_logit: (B,1) or (B,)
    """
    B = logits.shape[0]
    lab = label.view(B).float().to(logits.device)
    with torch.no_grad():
        pres = torch.sigmoid(cls_logit.view(B))
        tmax = p_teacher.view(B, -1).max(1).values
        pfrac = torch.sigmoid(logits).view(B, -1).mean(1)
        gate = (lab > 0.5) & (pres > 0.5) & (tmax > teacher_thr) & (pfrac < rho_lo)
    if gate.sum() < 1:
        return logits.new_zeros(())
    flat = logits.view(B, -1)[gate]
    kk = min(k, flat.shape[1])
    top = flat.topk(kk, dim=1).values
    conf = tmax[gate].unsqueeze(1)                      # trust teacher-confident slices more
    bce = F.binary_cross_entropy_with_logits(top, torch.ones_like(top), reduction='none')
    return (bce * conf).mean()


def size_floor_loss(logits, label, rho_lo):
    """Soft lower bound on the predicted tumor fraction for POSITIVE slices.

    rho = mean(sigmoid(logits)) per slice; penalise rho < rho_lo with a smooth
    squared hinge (bounded, no log blow-up). rho_lo comes from the federated size
    prior. Only pushes UP where the prediction is too small, so it cannot inflate
    slices that are already adequately/over-segmented (precision-safe).

    logits: (B,1,H,W)   label: (B,)   rho_lo: float
    """
    B = logits.shape[0]
    lab = label.view(B).float().to(logits.device)
    m = lab > 0.5
    if m.sum() < 1 or rho_lo <= 0:
        return logits.new_zeros(())
    rho = torch.sigmoid(logits).view(B, -1).mean(1)[m]   # per-slice fg fraction
    # RELATIVE deficit in [0,1] (absolute fraction deficits ~1e-2 would square to
    # a negligible, gradient-free term); 1 when empty, 0 once rho reaches the floor.
    deficit = F.relu(1.0 - rho / (rho_lo + EPS))
    return (deficit ** 2).mean()


@torch.no_grad()
def fg_fraction_stat(logits, label, fire_thr=0.5, min_frac=1e-4):
    """Per-batch robust statistic for the FEDERATED size prior (GT-free).

    For positive slices where the model already fires (max prob > fire_thr), the
    predicted tumor fraction is a proxy for the true disease size. We return the
    list of such per-slice fractions; the trainer accumulates them across the
    epoch and takes a median, and the server aggregates client medians into the
    global prior pi. Returns a 1-D tensor (possibly empty).
    """
    B = logits.shape[0]
    lab = label.view(B).float().to(logits.device)
    p = torch.sigmoid(logits).view(B, -1)
    fires = (p.max(1).values > fire_thr) & (lab > 0.5)
    if fires.sum() < 1:
        return logits.new_zeros(0)
    frac = p[fires].mean(1)
    return frac[frac > min_frac]


def _gated_crf_one(prob, intensity, kernel, dilation, sigma_int, sigma_xy):
    """Single-scale intensity-gated pairwise affinity at a given dilation.

    Neighbours are sampled on a dilated grid (range ~ dilation*(kernel//2)); the
    spatial weight uses the grid offset (scale-shared) and the intensity weight
    decides growth, so larger dilations carry genuine long-range affinity.
    """
    B, _, H, W = prob.shape
    pad = (kernel // 2) * dilation
    prob_u = F.unfold(prob, kernel, dilation=dilation, padding=pad).view(B, kernel * kernel, H, W)
    int_u = F.unfold(intensity, kernel, dilation=dilation, padding=pad).view(B, kernel * kernel, H, W)

    ar = torch.arange(kernel, device=prob.device, dtype=prob.dtype) - (kernel // 2)
    dy, dx = torch.meshgrid(ar, ar, indexing='ij')
    d2 = (dy ** 2 + dx ** 2).reshape(-1)
    w_xy = torch.exp(-d2 / (2 * sigma_xy ** 2)).view(1, -1, 1, 1)
    w_int = torch.exp(-((int_u - intensity) ** 2) / (2 * sigma_int ** 2))
    w = w_xy * w_int
    diff = (prob_u - prob).abs()
    return ((w * diff).sum(1) / (w.sum(1) + EPS)).mean()


def multiscale_gated_crf_loss(prob, image, kernel=3, dilations=(1, 2, 4),
                              sigma_int=0.15, sigma_xy=6.0):
    """Mean of intensity-gated affinity over several dilations (multi-range).

    prob: (B,1,H,W) in [0,1]   image: (B,Cin,H,W) input modalities
    """
    intensity = image.mean(1, keepdim=True)
    total = prob.new_zeros(())
    for d in dilations:
        total = total + _gated_crf_one(prob, intensity, kernel, d, sigma_int, sigma_xy)
    return total / len(dilations)

@torch.no_grad()
def healthy_feat_pool(feat, label):
    """Mean backbone feature over NEGATIVE-slice pixels (clean tumor-free).
    feat:(B,C,H,W) label:(B,). Returns (vec[C], count) or None."""
    B, C, H, W = feat.shape
    lab = label.view(B).float().to(feat.device)
    neg = lab < 0.5
    if int(neg.sum()) < 1:
        return None
    vec = feat[neg].mean(dim=(0, 2, 3))            # (C,)
    cnt = float(int(neg.sum()) * H * W)
    return vec.detach(), cnt


def healthy_distill_loss(feat, label, mu_healthy):
    """Pull negative-slice pixel features toward the global healthy prototype
    (cosine). Only negative slices contribute, so positives/foreground learning is
    untouched. mu_healthy: (C,) global consensus, or None (warm-up -> 0)."""
    if mu_healthy is None:
        return feat.new_zeros(())
    B, C, H, W = feat.shape
    lab = label.view(B).float().to(feat.device)
    neg = lab < 0.5
    if int(neg.sum()) < 1:
        return feat.new_zeros(())
    fhat = F.normalize(feat[neg], dim=1)
    mh = F.normalize(mu_healthy, dim=0).view(1, -1, 1, 1)
    cos = (fhat * mh).sum(1)                        # (n_neg,H,W)
    return (1.0 - cos).mean()
