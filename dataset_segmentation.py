
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


def _dedup_modalities(config, client_id):
    seen = set()
    mods = []
    for m in config['clients'][str(client_id)]:
        if m not in seen:
            seen.add(m)
            mods.append(m)
    return mods


def _merge_modalities(channels):
    """Same RGB merge convention as the rest of the pipeline."""
    if len(channels) == 1:
        return Image.merge("RGB", (channels[0], channels[0], channels[0]))
    elif len(channels) == 2:
        return Image.merge("RGB", (channels[0], channels[1], channels[1]))
    else:
        return Image.merge("RGB", (channels[0], channels[1], channels[2]))


class SegTrainDataset(Dataset):
    """Train on CAM pseudo-labels read from the manifest produced by
    save_pseudo_labels_binary.py.

    The manifest already encodes the image-level label correction (label==0 ->
    empty mask) and the blank gating, so the saved pseudo-mask is used as-is.
    """

    def __init__(self, manifest_df, patch_size, config, client_id):
        self.df = manifest_df.reset_index(drop=True)
        self.config = config
        self.client_id = client_id
        self.modalities = _dedup_modalities(config, client_id)
        self.num_modalities = len(self.modalities)
        self.transform = transforms.Compose([
            transforms.CenterCrop(patch_size),
            transforms.ToTensor(),
        ])
        # Mask is resized with nearest to stay binary.
        self.mask_transform = transforms.Compose([
            transforms.CenterCrop(patch_size),
            transforms.PILToTensor(),
        ])

    def __len__(self):
        return len(self.df)

    def _load_image(self, image_path):
        path_map = {
            'flair': image_path,
            't1ce': image_path.replace("flair", "t1ce"),
            't2': image_path.replace("flair", "t2"),
        }
        channels = [Image.open(path_map[m]).convert("L") for m in self.modalities]
        return _merge_modalities(channels)

    def __getitem__(self, index):
        row = self.df.iloc[index]
        image = self._load_image(str(row['image_path']))

        # Detect blank input (same rule used at inference / pseudo-label time).
        blank = 0
        for ch in image.split():
            if len(set(ch.getdata())) <= 1:
                blank = 1
                break

        img = self.transform(image)[:self.num_modalities]

        # Pseudo-mask: saved as 0/255 PNG -> {0,1}.
        pmask = Image.open(str(row['pseudo_mask_path'])).convert("L")
        mask = (self.mask_transform(pmask) > 127).float()  # (1, H, W)

        label = int(row['label'])
        if label == 0:
            mask = torch.zeros_like(mask)

        return img, mask, torch.tensor(label, dtype=torch.long), torch.tensor(blank, dtype=torch.long)


class SegValDataset(Dataset):
    """Validate against the ground-truth whole-tumor mask.

    Reads the original split CSV rows (image_path, mask_path, label, ...).
    The GT seg PNG is RGB; whole-tumor = any non-zero channel.
    """

    def __init__(self, split_df, patch_size, config, client_id):
        self.df = split_df.reset_index(drop=True)
        self.config = config
        self.client_id = client_id
        self.modalities = _dedup_modalities(config, client_id)
        self.num_modalities = len(self.modalities)
        self.transform = transforms.Compose([
            transforms.CenterCrop(patch_size),
            transforms.ToTensor(),
        ])
        self.mask_transform = transforms.Compose([
            transforms.CenterCrop(patch_size),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.df)

    def _load(self, image_path):
        path_map = {
            'flair': image_path,
            't1ce': image_path.replace("flair", "t1ce"),
            't2': image_path.replace("flair", "t2"),
        }
        channels = [Image.open(path_map[m]).convert("L") for m in self.modalities]
        image = _merge_modalities(channels)
        seg = Image.open(image_path.replace('flair', 'seg')).convert("RGB")
        return image, seg

    def __getitem__(self, index):
        row = self.df.iloc[index]
        image, seg = self._load(str(row.iloc[0]))  # column 0 == image_path

        blank = 0
        for ch in image.split():
            if len(set(ch.getdata())) <= 1:
                blank = 1
                break

        img = self.transform(image)[:self.num_modalities]

        # Whole-tumor GT: union over the 3 seg channels.
        seg_t = self.transform(seg)  # (3, H, W) in [0,1]
        gt = (seg_t.sum(dim=0, keepdim=True) > 0).float()  # (1, H, W)

        label = int(row.iloc[2])  # column 2 == label
        if label == 0:
            gt = torch.zeros_like(gt)

        return img, gt, torch.tensor(label, dtype=torch.long), torch.tensor(blank, dtype=torch.long)
