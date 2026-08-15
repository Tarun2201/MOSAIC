import os
import re
import numpy as np
import torch
from PIL import Image
from scipy import ndimage
from torch.utils.data import Dataset
from torchvision import transforms

from dataset_segmentation import _dedup_modalities, _merge_modalities  # reuse


def _despeckle(mask, min_px):
    """Drop connected components smaller than min_px (removes CAM speckle FP)."""
    lab, n = ndimage.label(mask)
    if n == 0:
        return mask
    keep = np.zeros_like(mask)
    for i in range(1, n + 1):
        comp = lab == i
        if comp.sum() >= min_px:
            keep |= comp
    return keep


def cam_trimap(cam, r_in=2, r_out=4, speckle_min=10):
    """Confident-learning trimap of a binary CAM (target denoising).

    Returns (mask_core, weight):
      * mask_core : confident FOREGROUND = erode(despeckle(cam), r_in)   (target=1)
      * weight    : 1 on confident-FG and confident-BG, 0 on the IGNORE ring
                    = dilate(cam,r_out) \\ erode(cam,r_in) (the noisy boundary).
    An EMPTY CAM (or one whose components are all speckle) -> whole slice IGNORE
    (weight all 0): we refuse to teach "no tumor" on a tumor-present slice.
    Nothing here ADDS foreground -- it only abstains on unreliable supervision.
    cam: bool (H,W). Returns two float32 (H,W).
    """
    cam = _despeckle(cam.astype(bool), speckle_min)
    if cam.sum() == 0:
        z = np.zeros(cam.shape, dtype=np.float32)
        return z, z.copy()                       # all-ignore
    core = ndimage.binary_erosion(cam, iterations=r_in)
    dil = ndimage.binary_dilation(cam, iterations=r_out)
    ignore = dil & ~core
    return core.astype(np.float32), (~ignore).astype(np.float32)


def _neighbor_flair_path(flair_path, offset):
    """Return the FLAIR path at slice (idx+offset), or the original if the
    neighbor file does not exist (volume boundary)."""
    if offset == 0:
        return flair_path
    m = re.search(r'^(.*_)(\d+)(\.png)$', flair_path)
    if not m:
        return flair_path
    idx = int(m.group(2)) + offset
    if idx < 0:
        return flair_path
    cand = f"{m.group(1)}{idx}{m.group(3)}"
    return cand if os.path.exists(cand) else flair_path


class _Base25D(Dataset):
    def __init__(self, df, patch_size, config, client_id, context=1):
        self.df = df.reset_index(drop=True)
        self.config = config
        self.client_id = client_id
        self.modalities = _dedup_modalities(config, client_id)
        self.num_modalities = len(self.modalities)
        self.offsets = list(range(-context, context + 1))
        self.transform = transforms.Compose([
            transforms.CenterCrop(patch_size),
            transforms.ToTensor(),
        ])

    def _merged_from_flair(self, flair_path):
        path_map = {
            'flair': flair_path,
            't1ce': flair_path.replace("flair", "t1ce"),
            't2': flair_path.replace("flair", "t2"),
        }
        channels = [Image.open(path_map[m]).convert("L") for m in self.modalities]
        return _merge_modalities(channels)

    def _load_stack(self, flair_path):
        slices = []
        blank = 0
        for d in self.offsets:
            npath = _neighbor_flair_path(flair_path, d)
            img = self._merged_from_flair(npath)
            if d == 0:  # blank flag from the center slice only
                for ch in img.split():
                    if len(set(ch.getdata())) <= 1:
                        blank = 1
                        break
            t = self.transform(img)[:self.num_modalities]  # (num_mod, H, W)
            slices.append(t)
        return torch.stack(slices, dim=0), blank  # (K, num_mod, H, W)


class SegTrainDataset25D(_Base25D):
    """Train: center-slice target is the CAM pseudo-mask."""

    def __init__(self, manifest_df, patch_size, config, client_id, context=1):
        super().__init__(manifest_df, patch_size, config, client_id, context)
        self.mask_transform = transforms.Compose([
            transforms.CenterCrop(patch_size),
            transforms.PILToTensor(),
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]
        stack, blank = self._load_stack(str(row['image_path']))
        pmask = Image.open(str(row['pseudo_mask_path'])).convert("L")
        mask = (self.mask_transform(pmask) > 127).float()
        label = int(row['label'])
        if label == 0:
            mask = torch.zeros_like(mask)
        return stack, mask, torch.tensor(label, dtype=torch.long), torch.tensor(blank, dtype=torch.long)


class SegTrainDataset25DTrimap(SegTrainDataset25D):
    """Train dataset: center-slice CAM is denoised into a confident-learning
    trimap. Returns (stack, mask, wmap, label, blank) -- the extra per-pixel
    weight map wmap is 0 on IGNORE pixels (noisy boundary + empty-CAM positive
    slices) and 1 elsewhere. Negative slices are untouched (all-BG, full weight),
    preserving the specificity anchor. See cam_trimap()."""

    def __init__(self, manifest_df, patch_size, config, client_id, context=1,
                 r_in=2, r_out=4, speckle_min=10, clean=True):
        super().__init__(manifest_df, patch_size, config, client_id, context)
        self.r_in = r_in; self.r_out = r_out; self.speckle_min = speckle_min
        self.clean = clean   # clean=False -> raw CAM + all-ones weight

    def __getitem__(self, index):
        row = self.df.iloc[index]
        stack, blank = self._load_stack(str(row['image_path']))
        pmask = Image.open(str(row['pseudo_mask_path'])).convert("L")
        cam = (self.mask_transform(pmask) > 127).squeeze(0).numpy()   # (H,W) bool
        label = int(row['label'])
        if label == 0:
            mask = torch.zeros(1, *cam.shape)
            wmap = torch.ones(1, *cam.shape)
        elif not self.clean:
            mask = torch.from_numpy(cam.astype('float32')).unsqueeze(0)   # raw CAM
            wmap = torch.ones(1, *cam.shape)
        else:
            mk, wm = cam_trimap(cam, self.r_in, self.r_out, self.speckle_min)
            mask = torch.from_numpy(mk).unsqueeze(0)
            wmap = torch.from_numpy(wm).unsqueeze(0)
        return stack, mask, wmap, torch.tensor(label, dtype=torch.long), torch.tensor(blank, dtype=torch.long)


class SegValDataset25D(_Base25D):
    """Val/Test: center-slice target is the GT whole-tumor mask."""

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]
        flair_path = str(row.iloc[0])
        stack, blank = self._load_stack(flair_path)
        seg = Image.open(flair_path.replace('flair', 'seg')).convert("RGB")
        seg_t = self.transform(seg)
        gt = (seg_t.sum(dim=0, keepdim=True) > 0).float()
        label = int(row.iloc[2])
        if label == 0:
            gt = torch.zeros_like(gt)
        return stack, gt, torch.tensor(label, dtype=torch.long), torch.tensor(blank, dtype=torch.long)
