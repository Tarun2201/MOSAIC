import torch
import torch.nn as nn
from PIL import Image
import argparse
import ast
import gc
from datetime import datetime
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import sys
from tqdm import tqdm
import copy
from time import sleep
import time
import torch.distributed as dist
import torch.multiprocessing as mp
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
import json
from torch.nn.parallel import DistributedDataParallel as DDP
from dataset_personalized_2 import InferenceDataset
from models_unet_scoring import *
from loss import *
from evaluation import *
from evaluation_3d import compute_3d_volume_metrics
from utils import *
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix


def _parse_volume_slice(img_name):
    """Parse a slice filename into (volume_id, slice_index).

    Example: 'FeTS2022_00000_flair_4' -> ('FeTS2022_00000', 4)
    """
    name = img_name
    if name.endswith('.png'):
        name = name[:-4]
    parts = name.split('_')
    try:
        slice_idx = int(parts[-1])
    except (ValueError, IndexError):
        slice_idx = 0
    vol_parts = parts[:-1]
    if vol_parts and vol_parts[-1] in ('flair', 't1ce', 't2', 't1', 'seg'):
        vol_parts = vol_parts[:-1]
    volume_id = '_'.join(vol_parts)
    return volume_id, slice_idx


torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
torch.backends.cudnn.deterministic = True


def _to_uint8_gray(arr):
    """Normalize a 2D float array to a uint8 (H,W) grayscale image."""
    arr = np.asarray(arr, dtype=np.float32)
    lo, hi = float(arr.min()), float(arr.max())
    if hi - lo < 1e-8:
        return np.zeros_like(arr, dtype=np.uint8)
    return ((arr - lo) / (hi - lo) * 255.0).astype(np.uint8)


def _cam_to_heatmap(cam):
    """Map a [0,1] CAM to a simple blue->red RGB heatmap (no matplotlib needed)."""
    cam = np.clip(np.asarray(cam, dtype=np.float32), 0, 1)
    r = np.clip(1.5 - np.abs(4 * cam - 3), 0, 1)
    g = np.clip(1.5 - np.abs(4 * cam - 2), 0, 1)
    b = np.clip(1.5 - np.abs(4 * cam - 1), 0, 1)
    return (np.stack([r, g, b], axis=-1) * 255).astype(np.uint8)


def _add_title(panel_rgb, text, color):
    from PIL import ImageDraw
    h, w, _ = panel_rgb.shape
    strip_h = 18
    img = Image.new("RGB", (w, h + strip_h), (0, 0, 0))
    img.paste(Image.fromarray(panel_rgb), (0, strip_h))
    ImageDraw.Draw(img).text((4, 3), text, fill=color)
    return np.array(img)


def save_cam_panel(flair, gt, cam, pred, dice, out_path):
    """Save a labeled 4-panel PNG:
        FLAIR (input) | GT (green) | CAM heatmap | Pred (red, thresholded).

    flair: (H,W) any-range float/uint8; gt/pred: (H,W) {0,1}; cam: (H,W) in [0,1].
    """
    flair_u8 = _to_uint8_gray(flair)
    base = np.stack([flair_u8, flair_u8, flair_u8], axis=-1)

    def overlay(mask, color):
        out = base.copy()
        m = mask.astype(bool)
        out[m] = (0.4 * out[m] + 0.6 * np.array(color)).astype(np.uint8)
        return out

    gt_color, pred_color = (0, 255, 0), (255, 0, 0)
    p_flair = _add_title(base, "FLAIR (input)", (255, 255, 255))
    p_gt = _add_title(overlay(gt, gt_color), "GT (green)", gt_color)
    p_cam = _add_title(_cam_to_heatmap(cam), "CAM map", (255, 255, 255))
    p_pred = _add_title(overlay(pred, pred_color), f"Pred (red) dice={dice:.3f}", pred_color)

    h = p_flair.shape[0]
    sep = np.full((h, 4, 3), 255, dtype=np.uint8)
    panel = np.concatenate([p_flair, sep, p_gt, sep, p_cam, sep, p_pred], axis=1)
    Image.fromarray(panel).save(out_path)


class Design_CAM_Binary(object):
    """Binary (whole-tumor) Dice evaluation.

    Mirrors the model loading / CAM extraction logic of the multilabel
    `Design_CAM` in test_agg_net_eval.py, but:
      * The prediction is the BINARY cam (`bin_ame_map`, i.e. `ame_map[-1]`),
        which is produced by `bin_score_model` and is the same map used to
        gate the multiclass core/edema cams.
      * The ground truth is the WHOLE-TUMOR mask (union of core + edema, i.e.
        the union of all three seg channels).
      * A single CAM threshold is grid-searched to maximize binary Dice.
    """

    def __init__(self, args, client_id, config):

        self.task = args.task
        self.device = args.device
        self.client_id = client_id
        # Number of genuine input modalities for this client (deduped, order-preserved),
        # used to detect a blank input modality at test time.
        _seen = set()
        self.num_modalities = 0
        for m in config['clients'][str(client_id)]:
            if m not in _seen:
                _seen.add(m)
                self.num_modalities += 1

        # Load client-specific models (UNet + Modality)
        multi_model = Res18_Classifier(num_classes=2)
        binary_model = Res18_Classifier(num_classes=1)
        bin_score_model = Res_Scoring(use_unet=True, spatial_normalize=getattr(args, 'spatial_normalize', False)).to(self.device)
        multi_score_model = Res_Scoring(use_unet=True, spatial_normalize=getattr(args, 'spatial_normalize', False)).to(self.device)

        # Soft gating alpha for test time
        self.soft_gating_alpha = getattr(args, 'soft_gating_alpha', 0.0)

        # Client-specific modality models
        binary_modality_model = SimpleUNet(in_channels=len(config['clients'][str(client_id)]))
        multi_modality_model = SimpleUNet(in_channels=len(config['clients'][str(client_id)]))
        bin_score_modality_model = SimpleUNet(in_channels=len(config['clients'][str(client_id)]))
        multi_score_modality_model = SimpleUNet(in_channels=len(config['clients'][str(client_id)]))

        # Load client-specific pretrained models
        if str(client_id) in args.bin_pretrained_paths:
            binary_model.load_pretrain_weight(args.bin_pretrained_paths[str(client_id)])
            print(f"Loaded binary UNet for client {client_id}")
        else:
            raise Exception(f"No pretrained binary UNet found for client {client_id}")

        if str(client_id) in args.multi_pretrained_paths:
            multi_model.load_pretrain_weight(args.multi_pretrained_paths[str(client_id)])
            print(f"Loaded multiclass UNet for client {client_id}")
        else:
            raise Exception(f"No pretrained multiclass UNet found for client {client_id}")

        if str(client_id) in args.bin_modality_pretrained_paths:
            checkpoint = torch.load(args.bin_modality_pretrained_paths[str(client_id)], map_location=self.device)
            binary_modality_model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Loaded binary modality model for client {client_id}")
        else:
            raise Exception(f"No pretrained binary modality model found for client {client_id}")

        if str(client_id) in args.multi_modality_pretrained_paths:
            checkpoint = torch.load(args.multi_modality_pretrained_paths[str(client_id)], map_location=self.device)
            multi_modality_model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Loaded multiclass modality model for client {client_id}")
        else:
            raise Exception(f"No pretrained multiclass modality model found for client {client_id}")

        # Binary score modality model
        if hasattr(args, 'bin_score_modality_pretrained_paths') and str(client_id) in args.bin_score_modality_pretrained_paths:
            checkpoint = torch.load(args.bin_score_modality_pretrained_paths[str(client_id)], map_location=self.device)
            bin_score_modality_model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Loaded binary score modality model for client {client_id}")
        else:
            print(f"Warning: No pretrained binary score modality model found for client {client_id}, using binary modality model instead")
            bin_score_modality_model = binary_modality_model

        # Multiclass score modality model
        if hasattr(args, 'multi_score_modality_pretrained_paths') and str(client_id) in args.multi_score_modality_pretrained_paths:
            checkpoint = torch.load(args.multi_score_modality_pretrained_paths[str(client_id)], map_location=self.device)
            multi_score_modality_model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Loaded multiclass score modality model for client {client_id}")
        else:
            print(f"Warning: No pretrained multiclass score modality model found for client {client_id}, using multiclass modality model instead")
            multi_score_modality_model = multi_modality_model

        # Scoring models
        if hasattr(args, 'bin_score_model_pretrained_paths') and str(client_id) in args.bin_score_model_pretrained_paths:
            bin_score_model.load_pretrain_weight(args.bin_score_model_pretrained_paths[str(client_id)])
            print(f"Loaded binary scoring model for client {client_id}")
        elif hasattr(args, 'bin_score_model_pretrained_path'):
            bin_score_model.load_pretrain_weight(args.bin_score_model_pretrained_path)
            print(f"Loaded global binary scoring model")

        if hasattr(args, 'multi_score_model_pretrained_paths') and str(client_id) in args.multi_score_model_pretrained_paths:
            multi_score_model.load_pretrain_weight(args.multi_score_model_pretrained_paths[str(client_id)])
            print(f"Loaded multiclass scoring model for client {client_id}")
        elif hasattr(args, 'multi_score_model_pretrained_path'):
            multi_score_model.load_pretrain_weight(args.multi_score_model_pretrained_path)
            print(f"Loaded global multiclass scoring model")

        # Freeze all models
        for m in [binary_model, multi_model, binary_modality_model, multi_modality_model,
                  bin_score_model, multi_score_model, bin_score_modality_model, multi_score_modality_model]:
            for param in m.parameters():
                param.requires_grad = False

        self.binary_model = binary_model.to(self.device).eval()
        self.multi_model = multi_model.to(self.device).eval()
        self.binary_modality_model = binary_modality_model.to(self.device).eval()
        self.multi_modality_model = multi_modality_model.to(self.device).eval()
        self.bin_score_model = bin_score_model.to(self.device).eval()
        self.multi_score_model = multi_score_model.to(self.device).eval()
        self.bin_score_modality_model = bin_score_modality_model.to(self.device).eval()
        self.multi_score_modality_model = multi_score_modality_model.to(self.device).eval()
        self.freq_band = str(args.freq_band)

        self.save_dir = f'./results/eval_binary/freq_band_{self.freq_band}/client_{client_id}'
        os.makedirs(self.save_dir, exist_ok=True)

    def step(self, img):
        """Identical forward pass to the multilabel evaluator.

        Returns the full attention map (multiclass core/edema cams concatenated
        with the binary cam as the last channel), the binary logit and the
        multiclass logits. The binary cam == ame_map[:, -1].
        """
        img = img.to(self.device)

        # --- Pass through client-specific modality models first ---
        binary_modality_features = self.binary_modality_model(img)
        multi_modality_features = self.multi_modality_model(img)

        # --- Binary and Multi CAM extraction (using modality features) ---
        logits_collect_binary, map_collect_binary = self.binary_model(binary_modality_features)
        logits_collect_multi, map_collect_multi = self.multi_model(multi_modality_features)

        # --- Binary ame map computation (using bin_score_modality_model) ---
        bin_score_modality_features = self.bin_score_modality_model(img)
        map_collect_binary_copy = [t.clone() for t in map_collect_binary]
        _, _, _, bin_ame_map = self.bin_score_model(bin_score_modality_features, map_collect_binary_copy)
        bin_ame_map = self.normalize_map(bin_ame_map)

        # --- binary guidance ---
        map_collect_multi = torch.stack(map_collect_multi, dim=0)
        if self.soft_gating_alpha > 0:
            map_collect = (self.soft_gating_alpha + (1 - self.soft_gating_alpha) * bin_ame_map) * map_collect_multi
        else:
            map_collect = bin_ame_map * map_collect_multi

        map_collect = list(map_collect.unbind(0))

        # --- Multi-class score model (using multi_score_modality_model) ---
        multi_score_modality_features = self.multi_score_modality_model(img)
        _, _, _, ame_map = self.multi_score_model(multi_score_modality_features, map_collect)

        # --- Final attention maps (binary cam appended as last channel) ---
        ame_map = torch.cat((ame_map, bin_ame_map), dim=1)

        return ame_map.detach().cpu(), logits_collect_binary[-1].detach().cpu(), logits_collect_multi[-1].detach().cpu()

    @staticmethod
    def _has_blank_modality(case_batch, num_modalities, eps=1e-6):
        """Return True if ANY input modality channel of the slice is blank.

        `case_batch` is the model-input tensor of shape (1, C, H, W) where the
        first `num_modalities` channels are the real per-client modalities (the
        remaining channels, if any, are duplicates from the RGB merge). A
        modality is considered blank when its pixels are (near-)uniform, i.e.
        it carries no signal (empty/missing slice for that modality).
        """
        img = case_batch
        if img.dim() == 3:  # (C, H, W) -> (1, C, H, W)
            img = img.unsqueeze(0)
        # Only inspect the genuine modality channels, not RGB-merge duplicates.
        chans = img[0, :num_modalities]
        for c in range(chans.shape[0]):
            ch = chans[c]
            if (ch.max() - ch.min()) <= eps:
                return True
        return False

    def normalize_map(self, att_map: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
        n, c, h, w = att_map.size()
        flat = att_map.view(n, c, -1)
        min_val = flat.min(2, keepdim=True)[0]
        max_val = flat.max(2, keepdim=True)[0]
        normalized = (flat - min_val) / (max_val - min_val + eps)
        return normalized.view(n, c, h, w)

    def postprocess_cam(self, binary_cam, cam_threshold=None):
        """Threshold the (already normalized) binary cam to a binary mask."""
        binary_cam = np.maximum(binary_cam, 0)
        binary_cam = binary_cam / (binary_cam.max() + 1e-8)

        if cam_threshold is None:
            cam_threshold = 0.5

        seg = np.where(binary_cam > cam_threshold, 1, 0)
        return seg

    def CAM_algo(self, input_image, ame_map, img_name, output_hist=False):
        for i in range(ame_map.shape[1]):
            if (ame_map[0][i].max() - ame_map[0][i].min()) > 0:
                ame_map[0][i] = (ame_map[0][i] - ame_map[0][i].min()) / (ame_map[0][i].max() - ame_map[0][i].min() + 1e-5)

        ame_map = ame_map.squeeze(0).numpy()
        # Per-channel adaptive inversion: invert only if mean > 0.5 (background is activated)
        for i in range(ame_map.shape[0]):
            if ame_map[i].mean() > 0.5:
                ame_map[i] = 1 - ame_map[i]
        return ame_map

    @staticmethod
    def _whole_tumor_gt(seg_batch):
        """Union of all seg channels -> whole tumor (binary) ground truth."""
        gt = (np.where(seg_batch[0][0].numpy() != 0, 1, 0)
              + np.where(seg_batch[0][1].numpy() != 0, 1, 0)
              + np.where(seg_batch[0][2].numpy() != 0, 1, 0))
        return np.clip(gt, 0, 1)

    def find_best_threshold(self, loader):
        """Grid-search a single CAM threshold to maximize whole-tumor Dice."""
        val_data = []  # list of (binary_cam, gt, binary_logit, type_label)

        print("  Collecting binary CAM data for Dice-based threshold search...")
        with torch.no_grad():
            for type_label, img_name, case_batch, seg_batch in tqdm(loader, desc="  Processing"):
                ame_map, binary_logit, class_logit = self.step(case_batch)
                binary_logit = binary_logit.squeeze(0).cpu().numpy()

                input_image = case_batch[0]
                ame_map = self.CAM_algo(input_image, ame_map, img_name)

                gt = self._whole_tumor_gt(seg_batch)

                binary_cam = np.maximum(ame_map[-1], 0)
                binary_cam = binary_cam / (binary_cam.max() + 1e-8)

                val_data.append({
                    'binary_cam': binary_cam,
                    'gt': gt,
                    'logit': float(binary_logit.reshape(-1)[0]),
                    'type_label': type_label[0],
                    'blank_modality': self._has_blank_modality(case_batch, self.num_modalities),
                })

        def compute_dice(pred, gt):
            intersection = np.sum(pred * gt)
            union = np.sum(pred) + np.sum(gt)
            if union == 0:
                return 1.0 if np.sum(gt) == 0 else 0.0
            return 2.0 * intersection / union

        def evaluate_threshold(cam_thresh):
            dice_scores = []
            for item in val_data:
                if item['logit'] < 0.5 or item['type_label'] == "blank" or item['blank_modality']:
                    pred = np.zeros_like(item['gt'])
                else:
                    pred = np.where(item['binary_cam'] > cam_thresh, 1, 0)
                dice_scores.append(compute_dice(pred, item['gt']))
            return np.mean(dice_scores)

        print("  Grid searching for optimal binary threshold (maximizing Dice)...")
        best_dice = -1
        best_thresh = 0.5
        for cam_t in np.arange(0.1, 0.9, 0.05):
            dice = evaluate_threshold(cam_t)
            if dice > best_dice:
                best_dice = dice
                best_thresh = cam_t

        # Fine-tune
        for cam_t in np.arange(max(0.05, best_thresh - 0.1), min(0.95, best_thresh + 0.1), 0.01):
            dice = evaluate_threshold(cam_t)
            if dice > best_dice:
                best_dice = dice
                best_thresh = cam_t

        print(f"    Binary: Best Dice={best_dice:.4f} at cam_thresh={best_thresh:.3f}")
        return best_thresh

    def run_tumor_test(self, loader, threshold=None, eval_3d=False,
                       save_top_k=0, vis_dir=None, flair_paths=None):
        self.binary_model.eval()
        self.multi_model.eval()
        self.bin_score_model.eval()
        self.multi_score_model.eval()

        if threshold is None:
            threshold = 0.5

        print(f"Using binary CAM threshold: {threshold:.3f}\n")

        log_path = os.path.join(self.save_dir, "results_binary.log")
        with open(log_path, "w+") as log_file:
            log_file.writelines(str(datetime.now()) + "\n")
            log_file.writelines(f"Binary CAM threshold: {threshold:.3f}\n")

        csv_path = os.path.join(self.save_dir, "tumor_result_binary.csv")
        with open(csv_path, "w+") as csv_file:
            csv_file.writelines("Img Name, Binary Dice, Binary IoU, Binary HD95\n")

        test_bar = tqdm(loader)
        result_metric = {'Binary Dice': [], 'Binary IoU': [], 'Binary HD95': []}

        # Per-volume accumulator for optional 3D evaluation:
        # volume_data[vol_id] = list of (slice_idx, gt_slice, pred_slice)
        volume_data = {}
        # Candidates for top-K CAM visualization:
        # (dice, flair_disp, gt, cam, pred, img_name)
        top_candidates = []

        with torch.no_grad():
            for sample_idx, (type, img_name, case_batch, seg_batch) in enumerate(test_bar):
                img_name = img_name[0][:-4]

                ame_map, binary_logit, class_logit = self.step(case_batch)
                binary_logit = binary_logit.squeeze(0).cpu().numpy()

                input_image = case_batch[0]
                ame_map = self.CAM_algo(input_image, ame_map, img_name)

                gt = self._whole_tumor_gt(seg_batch)

                # Continuous binary CAM (normalized to [0,1]) before thresholding.
                binary_cam = np.maximum(ame_map[-1], 0)
                binary_cam = binary_cam / (binary_cam.max() + 1e-8)

                final_seg = self.postprocess_cam(ame_map[-1], threshold)

                # use predicted logit to determine no tumor / tumor
                forced_blank = False
                if float(binary_logit.reshape(-1)[0]) < 0.5:
                    final_seg = np.zeros_like(gt)
                    forced_blank = True
                if type[0] == "blank":
                    final_seg = np.zeros_like(gt)
                    forced_blank = True
                # If ANY input modality for this slice is blank, force a blank prediction.
                if self._has_blank_modality(case_batch, self.num_modalities):
                    final_seg = np.zeros_like(gt)
                    forced_blank = True

                result = compute_seg_metrics(gt, final_seg)

                # Collect for top-K CAM visualization (skip perfect/blank).
                if save_top_k > 0 and result['Dice'] < 1.0 and not forced_blank:
                    # Use the on-disk FLAIR if available, else channel 0 of input.
                    if flair_paths is not None and sample_idx < len(flair_paths):
                        try:
                            fl = np.array(Image.open(flair_paths[sample_idx]).convert("L"))
                            from torchvision import transforms as _T
                            fl = np.array(_T.CenterCrop(gt.shape[-1])(Image.fromarray(fl)))
                            flair_disp = fl
                        except Exception:
                            flair_disp = input_image[0].cpu().numpy()
                    else:
                        flair_disp = input_image[0].cpu().numpy()
                    cam_thresholded = (binary_cam > threshold).astype(np.uint8)
                    top_candidates.append(
                        (result['Dice'], flair_disp, gt.astype(np.uint8),
                         binary_cam.copy(), final_seg.astype(np.uint8), img_name))

                with open(csv_path, "a") as csv_file:
                    csv_file.writelines(
                        f"{img_name}, {result['Dice']:.3f}, {result['IoU']:.3f}, {result['HD95']:.3f}\n"
                    )

                result_metric['Binary Dice'].append(result['Dice'])
                result_metric['Binary IoU'].append(result['IoU'])
                result_metric['Binary HD95'].append(result['HD95'])

                if eval_3d:
                    vol_id, slice_idx = _parse_volume_slice(img_name)
                    volume_data.setdefault(vol_id, []).append(
                        (slice_idx, gt.astype(np.uint8), final_seg.astype(np.uint8))
                    )

        # ---- save top-K highest-Dice CAM panels ----
        if save_top_k > 0 and top_candidates:
            top_candidates.sort(key=lambda x: x[0], reverse=True)
            client_vis_dir = os.path.join(vis_dir, f"client_{self.client_id}")
            os.makedirs(client_vis_dir, exist_ok=True)
            for rank, (dice, flair_disp, gt_np, cam_np, pred_np, name) in enumerate(
                    top_candidates[:save_top_k], 1):
                out_path = os.path.join(client_vis_dir, f"rank{rank:02d}_dice{dice:.3f}_{name}.png")
                save_cam_panel(flair_disp, gt_np, cam_np, pred_np, dice, out_path)
            print(f"  Saved top {min(save_top_k, len(top_candidates))} CAM panels -> {client_vis_dir}")

        for k, v in result_metric.items():
            result_metric[k] = np.mean(v)
        test_bar.close()

        with open(log_path, "a") as log_file:
            log_file.writelines("Average Results (2D, per-slice)\n")
            for k, v in result_metric.items():
                log_file.writelines(f"{k}: {v:.3f}\n")

        if eval_3d:
            metrics_3d = self._compute_3d_metrics(volume_data, log_path)
            result_metric.update(metrics_3d)

        return result_metric

    def _compute_3d_metrics(self, volume_data, log_path):
        """Stack per-slice binary predictions into (D, H, W) volumes and compute 3D metrics."""
        csv3d_path = os.path.join(self.save_dir, "tumor_result_binary_3d.csv")
        with open(csv3d_path, "w+") as f:
            f.writelines("Volume, Binary Dice, Binary IoU, Binary HD95\n")

        agg = {'Dice': [], 'IoU': [], 'HD95': []}

        print(f"\nComputing 3D volumetric metrics over {len(volume_data)} volumes...")
        for vol_id in tqdm(sorted(volume_data.keys()), desc="  3D volumes"):
            slices = sorted(volume_data[vol_id], key=lambda x: x[0])
            if len(slices) == 0:
                continue
            gt_vol = np.stack([s[1] for s in slices], axis=0)
            pred_vol = np.stack([s[2] for s in slices], axis=0)
            m = compute_3d_volume_metrics(gt_vol, pred_vol)
            agg['Dice'].append(m['Dice'])
            agg['IoU'].append(m['IoU'])
            agg['HD95'].append(m['HD95'])
            with open(csv3d_path, "a") as f:
                f.writelines(f"{vol_id}, {m['Dice']:.3f}, {m['IoU']:.3f}, {m['HD95']:.3f}\n")

        metrics_3d = {
            '3D Binary Dice': float(np.mean(agg['Dice'])) if agg['Dice'] else float('nan'),
            '3D Binary IoU': float(np.mean(agg['IoU'])) if agg['IoU'] else float('nan'),
            '3D Binary HD95': float(np.mean(agg['HD95'])) if agg['HD95'] else float('nan'),
        }

        with open(log_path, "a") as log_file:
            log_file.writelines("\nAverage Results (3D, per-volume)\n")
            for k, v in metrics_3d.items():
                log_file.writelines(f"{k}: {v:.3f}\n")

        return metrics_3d


def evaluation(method="FedAvg", model=None, csv_path=None, client_list=None, num_rounds=200, global_model=None, device=0, img_size=None, checkpoint=None, batch_size=8, num_workers=4, temperature=0.07, patience=20, starting_lr=1e-3, task='binary',
               bin_pretrained_dir=None, bin_modality_pretrained_dir=None,
               multi_pretrained_dir=None, multi_modality_pretrained_dir=None,
               bin_score_pretrained_dir=None, multi_score_pretrained_dir=None,
               bin_score_modality_pretrained_dir=None, multi_score_modality_pretrained_dir=None, freq_band=8,
               spatial_normalize=False, soft_gating_alpha=0.0, eval_3d=False,
               save_top_k=0, vis_dir='top_slices_cam_vis'):

    # Setup arguments for Design_CAM_Binary
    agg_net_args = argparse.Namespace()
    agg_net_args.epochs = 5
    agg_net_args.batch_size = 128
    agg_net_args.num_classes = 2
    agg_net_args.task = "binary" if task == 'binary' else "multiclass"
    agg_net_args.learning_rate = 0.0005
    agg_net_args.device = device
    agg_net_args.loss_weight = [1.0, 1.0, 1.0, 1.0, 5.0]
    agg_net_args.spatial_normalize = spatial_normalize
    agg_net_args.soft_gating_alpha = soft_gating_alpha

    # Build client-specific pretrained model paths
    agg_net_args.bin_pretrained_paths = {}
    agg_net_args.bin_modality_pretrained_paths = {}
    agg_net_args.multi_pretrained_paths = {}
    agg_net_args.multi_modality_pretrained_paths = {}
    agg_net_args.bin_score_model_pretrained_paths = {}
    agg_net_args.multi_score_model_pretrained_paths = {}
    agg_net_args.bin_score_modality_pretrained_paths = {}
    agg_net_args.multi_score_modality_pretrained_paths = {}
    agg_net_args.freq_band = freq_band

    for client in client_list:
        if bin_pretrained_dir:
            agg_net_args.bin_pretrained_paths[str(client)] = os.path.join(
                bin_pretrained_dir, f"_personalized_unet_client_{client}.pth")
        if bin_modality_pretrained_dir:
            agg_net_args.bin_modality_pretrained_paths[str(client)] = os.path.join(
                bin_modality_pretrained_dir, f"_personalized_modality_client_{client}.pth")
        if multi_pretrained_dir:
            agg_net_args.multi_pretrained_paths[str(client)] = os.path.join(
                multi_pretrained_dir, f"_personalized_unet_client_{client}.pth")
        if multi_modality_pretrained_dir:
            agg_net_args.multi_modality_pretrained_paths[str(client)] = os.path.join(
                multi_modality_pretrained_dir, f"_personalized_modality_client_{client}.pth")
        if bin_score_pretrained_dir:
            agg_net_args.bin_score_model_pretrained_paths[str(client)] = os.path.join(
                bin_score_pretrained_dir, f"_personalized_scoring_client_{client}_best.pth")
        if multi_score_pretrained_dir:
            agg_net_args.multi_score_model_pretrained_paths[str(client)] = os.path.join(
                multi_score_pretrained_dir, f"_personalized_scoring_client_{client}_best.pth")
        if bin_score_modality_pretrained_dir:
            agg_net_args.bin_score_modality_pretrained_paths[str(client)] = os.path.join(
                bin_score_modality_pretrained_dir, f"_personalized_modality_client_{client}_best.pth")
        if multi_score_modality_pretrained_dir:
            agg_net_args.multi_score_modality_pretrained_paths[str(client)] = os.path.join(
                multi_score_modality_pretrained_dir, f"_personalized_modality_client_{client}_best.pth")

    df = pd.read_csv(csv_path)
    if task == 'binary':
        config = {
            'dataset': 'brats',
            'task': 'binary',
            'combine': None,
            'clients': {
                "1": ["flair", "t1ce"],
                "2": ["flair", "t2"],
                "3": ["t1ce", "t2"],
                "4": ["flair"],
                "5": ["t1ce", "t2"],
                "6": ["flair", "t1ce"],
            }
        }
    else:
        config = {
            'dataset': 'brats',
            'task': 'multiclass',
            'combine': {
                'core': ['necrosis', 'enhancing'],
                'edema': ['edema']
            },
            'clients': {
                "1": ["flair", "t1ce"],
                "2": ["flair", "t2"],
                "3": ["t1ce", "t2"],
                "4": ["flair"],
                "5": ["t1ce", "t2"],
                "6": ["flair", "t1ce"],
            }
        }

    all_results = {}
    for client in client_list:
        print(f"\n{'='*60}")
        print(f"Evaluating Client {client} (Binary whole-tumor Dice)")
        print(f"{'='*60}")

        test_df = df[(df['split'] == "test") & (df['client_id'] == client)].reset_index(drop=True)
        val_df = df[((df['split'] == "val")) & (df['client_id'] == client)].reset_index(drop=True)

        if len(test_df) == 0:
            print(f"Warning: No test data found for client {client}")
            continue

        des_cam = Design_CAM_Binary(agg_net_args, client_id=client, config=config)

        # Threshold search on validation data
        val_dataset = InferenceDataset(val_df, img_size, config, client_id=client)
        val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=num_workers, pin_memory=True)
        threshold = des_cam.find_best_threshold(val_dataloader)

        # Evaluate on test data
        test_dataset = InferenceDataset(test_df, img_size, config, client_id=client)
        test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=num_workers, pin_memory=True)
        # FLAIR paths aligned to loader order (shuffle=False) for the input panel.
        flair_paths = test_df.iloc[:, 0].astype(str).tolist()
        result_metrics = des_cam.run_tumor_test(
            test_dataloader, threshold=threshold, eval_3d=eval_3d,
            save_top_k=save_top_k, vis_dir=vis_dir, flair_paths=flair_paths)
        all_results[client] = result_metrics

        print(f"\nClient {client} Results:")
        for k, v in result_metrics.items():
            print(f"  {k}: {v:.3f}")

    if all_results:
        print(f"\n{'='*60}")
        print("Average Binary Results Across All Clients")
        print(f"{'='*60}")
        for metric in list(all_results.values())[0].keys():
            avg = np.mean([result[metric] for result in all_results.values()])
            print(f"{metric}: {avg:.3f}")

    return global_model


if (__name__ == "__main__"):
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', type=str, default="FedAvg", help='Federated learning method')
    parser.add_argument('--csv_path', type=str, required=True, help='Path to the CSV file')
    parser.add_argument('--clients', type=str, default="[1,2,3,4,5]", help='List of client IDs')
    parser.add_argument('--num_rounds', type=int, default=200, help='Number of communication rounds')
    parser.add_argument('--img_size', type=str, default="[224]", help='Image size')
    parser.add_argument('--checkpoint', type=int, default=None, help='Path to checkpoint')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for training')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers for data loading')
    parser.add_argument('--temperature', type=float, default=0.07, help='Temperature for contrastive loss')
    parser.add_argument('--patience', type=int, default=20, help='Patience for early stopping')
    parser.add_argument('--starting_lr', type=float, default=1e-3, help='Starting learning rate for optimizers')
    parser.add_argument('--gpu', type=int, default=0, help='GPU id to use')
    parser.add_argument('--task', type=str, default='multiclass', help='Task type: binary or multiclass (models are multiclass aggnet)')

    parser.add_argument('--bin_pretrained_dir', type=str, default=None, help='Directory containing client-specific binary UNet models')
    parser.add_argument('--bin_modality_pretrained_dir', type=str, default=None, help='Directory containing client-specific binary modality models')
    parser.add_argument('--multi_pretrained_dir', type=str, default=None, help='Directory containing client-specific multiclass UNet models')
    parser.add_argument('--multi_modality_pretrained_dir', type=str, default=None, help='Directory containing client-specific multiclass modality models')
    parser.add_argument('--bin_score_pretrained_dir', type=str, default=None, help='Directory containing client-specific binary scoring models')
    parser.add_argument('--multi_score_pretrained_dir', type=str, default=None, help='Directory containing client-specific multiclass scoring models')
    parser.add_argument('--bin_score_modality_pretrained_dir', type=str, default=None, help='Directory containing client-specific binary score modality models')
    parser.add_argument('--multi_score_modality_pretrained_dir', type=str, default=None, help='Directory containing client-specific multiclass score modality models')
    parser.add_argument('--freq_band', type=int, default=8, help='Frequency band for ablation study')

    parser.add_argument('--spatial_normalize', action='store_true',
                        help='Use spatial normalization in Res_Scoring (must match training flag)')
    parser.add_argument('--soft_gating_alpha', type=float, default=0.0,
                        help='Leakage factor for soft binary gating (must match training value)')
    parser.add_argument('--eval_3d', action='store_true',
                        help='Also compute 3D volumetric Dice/IoU/HD95 (slices grouped by volume id)')
    parser.add_argument('--save_top_k', type=int, default=0,
                        help='Per client, save this many highest-Dice CAM panels (Dice==1/blank skipped).')
    parser.add_argument('--vis_dir', type=str, default='top_slices_cam_vis',
                        help='New folder in cwd for the FLAIR|GT|CAM|Pred panels.')
    args = parser.parse_args()

    method = args.method
    csv_path = args.csv_path
    client_list = ast.literal_eval(args.clients)
    num_rounds = args.num_rounds
    img_size = ast.literal_eval(args.img_size)
    checkpoint = args.checkpoint
    batch_size = args.batch_size
    num_workers = args.num_workers
    temperature = args.temperature
    patience = args.patience
    starting_lr = args.starting_lr
    device = "cuda:0"
    print("Using device: ", device)
    global_model = None
    model = None
    evaluation(method=method, model=model, csv_path=csv_path, client_list=client_list,
               num_rounds=num_rounds, global_model=global_model, device=device, img_size=img_size,
               checkpoint=checkpoint, batch_size=batch_size, num_workers=num_workers,
               temperature=temperature, patience=patience, starting_lr=starting_lr, task=args.task,
               bin_pretrained_dir=args.bin_pretrained_dir,
               bin_modality_pretrained_dir=args.bin_modality_pretrained_dir,
               multi_pretrained_dir=args.multi_pretrained_dir,
               multi_modality_pretrained_dir=args.multi_modality_pretrained_dir,
               bin_score_pretrained_dir=args.bin_score_pretrained_dir,
               multi_score_pretrained_dir=args.multi_score_pretrained_dir,
               bin_score_modality_pretrained_dir=args.bin_score_modality_pretrained_dir,
               multi_score_modality_pretrained_dir=args.multi_score_modality_pretrained_dir, freq_band=args.freq_band,
               spatial_normalize=args.spatial_normalize, soft_gating_alpha=args.soft_gating_alpha, eval_3d=args.eval_3d,
               save_top_k=args.save_top_k, vis_dir=args.vis_dir)
