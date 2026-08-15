"""Generate and save binary (whole-tumor) pseudo-labels for segmentation training.
See `run_save_pseudo_labels.sh` for an example invocation.
"""

import os
import ast
import argparse

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset_personalized_2 import InferenceDataset
from models_unet_scoring import Res18_Classifier, Res_Scoring, SimpleUNet


def build_config():
    """Client -> modality map used by the binary pipeline."""
    return {
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


class BinaryCAM(object):
    """Whole-tumor CAM extraction from the binary branch only.

    Loads four client-specific checkpoints:
      * binary_model            -- Res18_Classifier(num_classes=1), multi-scale CAMs
      * binary_modality_model   -- SimpleUNet, client's modalities -> 3 channels
      * bin_score_model         -- Res_Scoring, aggregates the 4 CAMs into one map
      * bin_score_modality_model-- SimpleUNet feeding the scoring net
                                   (falls back to binary_modality_model)

    `step` returns the aggregated binary CAM and the slice-level tumor logit.
    """

    def __init__(self, args, client_id, config):
        self.device = args.device
        self.client_id = client_id

        # Number of genuine input modalities for this client (deduped,
        # order-preserved), used to detect a blank input modality at test time.
        _seen = set()
        self.num_modalities = 0
        for m in config['clients'][str(client_id)]:
            if m not in _seen:
                _seen.add(m)
                self.num_modalities += 1

        in_ch = len(config['clients'][str(client_id)])
        c = str(client_id)

        # --- binary classifier (produces the 4 multi-scale CAMs) ---
        if c not in args.bin_pretrained_paths:
            raise Exception(f"No pretrained binary UNet found for client {client_id}")
        binary_model = Res18_Classifier(num_classes=1)
        binary_model.load_pretrain_weight(args.bin_pretrained_paths[c])
        print(f"Loaded binary UNet for client {client_id}")

        # --- binary modality model ---
        if c not in args.bin_modality_pretrained_paths:
            raise Exception(f"No pretrained binary modality model found for client {client_id}")
        binary_modality_model = SimpleUNet(in_channels=in_ch)
        ckpt = torch.load(args.bin_modality_pretrained_paths[c], map_location=self.device)
        binary_modality_model.load_state_dict(ckpt['model_state_dict'])
        print(f"Loaded binary modality model for client {client_id}")

        # --- binary scoring (aggregation) net ---
        if c not in args.bin_score_model_pretrained_paths:
            raise Exception(
                f"No pretrained binary scoring model found for client {client_id}. "
                "Pass --bin_score_pretrained_dir; without it the aggregation net "
                "would stay randomly initialised and the pseudo-labels would be garbage.")
        bin_score_model = Res_Scoring(
            use_unet=True, spatial_normalize=getattr(args, 'spatial_normalize', False))
        bin_score_model.load_pretrain_weight(args.bin_score_model_pretrained_paths[c])
        print(f"Loaded binary scoring model for client {client_id}")

        # --- modality model feeding the scoring net (optional) ---
        if c in args.bin_score_modality_pretrained_paths:
            bin_score_modality_model = SimpleUNet(in_channels=in_ch)
            ckpt = torch.load(args.bin_score_modality_pretrained_paths[c], map_location=self.device)
            bin_score_modality_model.load_state_dict(ckpt['model_state_dict'])
            print(f"Loaded binary score modality model for client {client_id}")
        else:
            print(f"Warning: No pretrained binary score modality model found for "
                  f"client {client_id}, using binary modality model instead")
            bin_score_modality_model = binary_modality_model

        for m in [binary_model, binary_modality_model, bin_score_model, bin_score_modality_model]:
            for param in m.parameters():
                param.requires_grad = False

        self.binary_model = binary_model.to(self.device).eval()
        self.binary_modality_model = binary_modality_model.to(self.device).eval()
        self.bin_score_model = bin_score_model.to(self.device).eval()
        self.bin_score_modality_model = bin_score_modality_model.to(self.device).eval()

    def step(self, img):
        """Forward pass -> (binary CAM (1,1,H,W), slice-level tumor logit)."""
        img = img.to(self.device)

        binary_modality_features = self.binary_modality_model(img)
        logits_collect_binary, map_collect_binary = self.binary_model(binary_modality_features)

        bin_score_modality_features = self.bin_score_modality_model(img)
        # Res_Scoring.forward appends to the list it is given, so pass a copy.
        map_collect_binary_copy = [t.clone() for t in map_collect_binary]
        _, _, _, bin_ame_map = self.bin_score_model(
            bin_score_modality_features, map_collect_binary_copy)
        bin_ame_map = self.normalize_map(bin_ame_map)

        return bin_ame_map.detach().cpu(), logits_collect_binary[-1].detach().cpu()

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

    def CAM_algo(self, ame_map):
        """Per-channel min-max normalize, then invert channels whose mean > 0.5
        (i.e. where the background rather than the tumor is activated)."""
        for i in range(ame_map.shape[1]):
            if (ame_map[0][i].max() - ame_map[0][i].min()) > 0:
                ame_map[0][i] = (ame_map[0][i] - ame_map[0][i].min()) / (
                    ame_map[0][i].max() - ame_map[0][i].min() + 1e-5)

        ame_map = ame_map.squeeze(0).numpy()
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
        val_data = []

        print("  Collecting binary CAM data for Dice-based threshold search...")
        with torch.no_grad():
            for type_label, img_name, case_batch, seg_batch in tqdm(loader, desc="  Processing"):
                ame_map, binary_logit = self.step(case_batch)
                binary_logit = binary_logit.squeeze(0).cpu().numpy()

                ame_map = self.CAM_algo(ame_map)
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


def build_pretrained_paths(args, client_list):
    """Resolve the --*_dir CLI args into per-client checkpoint file paths."""
    ns = argparse.Namespace()
    ns.device = args.device
    ns.spatial_normalize = args.spatial_normalize

    ns.bin_pretrained_paths = {}
    ns.bin_modality_pretrained_paths = {}
    ns.bin_score_model_pretrained_paths = {}
    ns.bin_score_modality_pretrained_paths = {}

    for client in client_list:
        c = str(client)
        if args.bin_pretrained_dir:
            ns.bin_pretrained_paths[c] = os.path.join(
                args.bin_pretrained_dir, f"_personalized_unet_client_{client}.pth")
        if args.bin_modality_pretrained_dir:
            ns.bin_modality_pretrained_paths[c] = os.path.join(
                args.bin_modality_pretrained_dir, f"_personalized_modality_client_{client}.pth")
        if args.bin_score_pretrained_dir:
            ns.bin_score_model_pretrained_paths[c] = os.path.join(
                args.bin_score_pretrained_dir, f"_personalized_scoring_client_{client}_best.pth")
        if args.bin_score_modality_pretrained_dir:
            ns.bin_score_modality_pretrained_paths[c] = os.path.join(
                args.bin_score_modality_pretrained_dir,
                f"_personalized_modality_client_{client}_best.pth")
    return ns


@torch.no_grad()
def save_split_pseudo_labels(des_cam, df_split, split_name, client, img_size,
                             config, num_workers, threshold, out_dir, manifest_rows):
    """Run the CAM pipeline over one (client, split) and write pseudo-mask PNGs.

    Appends one dict per slice to `manifest_rows`.
    """
    if len(df_split) == 0:
        print(f"  [client {client}] split '{split_name}' is empty, skipping.")
        return

    # We need the original dataframe row (for image_path / label) aligned with
    # the dataset order. InferenceDataset iterates df rows in order, so we keep
    # the reset-index df and index into it by position.
    df_split = df_split.reset_index(drop=True)
    dataset = InferenceDataset(df_split, img_size, config, client_id=client)
    loader = DataLoader(dataset, batch_size=1, shuffle=False,
                        num_workers=num_workers, pin_memory=True)

    split_out = os.path.join(out_dir, f"client_{client}", split_name)
    os.makedirs(split_out, exist_ok=True)

    for idx, (type_label, img_name, case_batch, seg_batch) in enumerate(
            tqdm(loader, desc=f"  client {client} / {split_name}")):
        img_name = img_name[0]
        stem = img_name[:-4] if img_name.endswith('.png') else img_name

        # --- image-level weak label for this slice (label column, index 2) ---
        image_label = int(df_split.iloc[idx, 2])
        image_path = str(df_split.iloc[idx, 0])
        gt_mask_path = str(df_split.iloc[idx, 1])

        # --- forward pass + CAM ---
        ame_map, binary_logit = des_cam.step(case_batch)
        binary_logit = binary_logit.squeeze(0).cpu().numpy()
        ame_map = des_cam.CAM_algo(ame_map)

        gt = des_cam._whole_tumor_gt(seg_batch)  # only for shape / reference

        pred = des_cam.postprocess_cam(ame_map[-1], threshold)

        # --- gating: empty prediction when uncertain / blank ---
        forced_blank = False
        if float(binary_logit.reshape(-1)[0]) < 0.5:
            pred = np.zeros_like(gt)
            forced_blank = True
        if type_label[0] == "blank":
            pred = np.zeros_like(gt)
            forced_blank = True
        if des_cam._has_blank_modality(case_batch, des_cam.num_modalities):
            pred = np.zeros_like(gt)
            forced_blank = True

        # --- IMAGE-LEVEL LABEL CORRECTION ---
        # If the weak label says no tumor, the pseudo-mask must be empty.
        corrected_by_label = False
        if image_label == 0 and pred.sum() > 0:
            pred = np.zeros_like(gt)
            corrected_by_label = True
        elif image_label == 0:
            # Already empty, but record that the label would have enforced it.
            corrected_by_label = True

        # --- write pseudo-mask PNG (0/255) ---
        mask_path = os.path.join(split_out, f"{stem}.png")
        Image.fromarray((pred.astype(np.uint8) * 255)).save(mask_path)

        manifest_rows.append({
            'image_path': image_path,
            'pseudo_mask_path': mask_path,
            'gt_mask_path': gt_mask_path,
            'label': image_label,
            'split': split_name,
            'client_id': client,
            'corrected_by_label': int(corrected_by_label),
            'forced_blank': int(forced_blank),
        })


def main():
    parser = argparse.ArgumentParser(
        description="Save binary (whole-tumor) pseudo-labels for segmentation training.")
    parser.add_argument('--clients', type=str, required=True,
                        help='List of client IDs, e.g. "[1,2,3,4]"')
    parser.add_argument('--csv_path', type=str, required=True)
    parser.add_argument('--img_size', type=str, default="[224]")
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--gpu', type=int, default=0)

    parser.add_argument('--bin_pretrained_dir', type=str, required=True,
                        help='Dir with _personalized_unet_client_<id>.pth (binary classifier)')
    parser.add_argument('--bin_modality_pretrained_dir', type=str, required=True,
                        help='Dir with _personalized_modality_client_<id>.pth')
    parser.add_argument('--bin_score_pretrained_dir', type=str, required=True,
                        help='Dir with _personalized_scoring_client_<id>_best.pth (agg net)')
    parser.add_argument('--bin_score_modality_pretrained_dir', type=str, default=None,
                        help='Dir with _personalized_modality_client_<id>_best.pth '
                             '(optional; falls back to --bin_modality_pretrained_dir)')

    parser.add_argument('--spatial_normalize', action='store_true',
                        help='Must match the flag used when training the agg net')

    parser.add_argument('--splits', type=str, default="train,val",
                        help='Comma-separated splits to generate pseudo-labels for.')
    parser.add_argument('--out_dir', type=str, default="./pseudo_labels_binary",
                        help='Root directory for pseudo-mask PNGs and manifest.csv')
    args = parser.parse_args()

    args.device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    print("Using device:", args.device)

    client_list = ast.literal_eval(args.clients)
    img_size = ast.literal_eval(args.img_size)
    splits = [s.strip() for s in args.splits.split(',') if s.strip()]

    config = build_config()
    cam_args = build_pretrained_paths(args, client_list)

    os.makedirs(args.out_dir, exist_ok=True)
    df = pd.read_csv(args.csv_path)

    manifest_rows = []
    for client in client_list:
        print(f"\n{'='*60}\nClient {client}\n{'='*60}")

        des_cam = BinaryCAM(cam_args, client_id=client, config=config)

        # Threshold searched on this client's val split.
        val_df = df[(df['split'] == "val") & (df['client_id'] == client)].reset_index(drop=True)
        if len(val_df) == 0:
            print(f"  Warning: no val data for client {client}; using threshold 0.5")
            threshold = 0.5
        else:
            val_dataset = InferenceDataset(val_df, img_size, config, client_id=client)
            val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False,
                                    num_workers=args.num_workers, pin_memory=True)
            threshold = des_cam.find_best_threshold(val_loader)
        print(f"  Using CAM threshold {threshold:.3f} for client {client}")

        for split_name in splits:
            df_split = df[(df['split'] == split_name) & (df['client_id'] == client)]
            save_split_pseudo_labels(
                des_cam, df_split, split_name, client, img_size, config,
                args.num_workers, threshold, args.out_dir, manifest_rows)

        del des_cam
        torch.cuda.empty_cache()

    manifest_path = os.path.join(args.out_dir, "manifest.csv")
    pd.DataFrame(manifest_rows).to_csv(manifest_path, index=False)

    # Quick summary.
    mdf = pd.DataFrame(manifest_rows)
    print(f"\n{'='*60}\nSaved {len(mdf)} pseudo-masks -> {args.out_dir}")
    print(f"Manifest: {manifest_path}")
    if len(mdf):
        print("Per split counts:")
        print(mdf.groupby('split').size())
        print(f"Corrected by image-level label (forced empty): "
              f"{int(mdf['corrected_by_label'].sum())}")
        print(f"Forced blank by logit/blank gating: {int(mdf['forced_blank'].sum())}")


if __name__ == "__main__":
    main()
