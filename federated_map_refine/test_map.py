
import os
import ast
import sys
import argparse
from datetime import datetime

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataset_segmentation import SegValDataset  # noqa: E402
from evaluation import compute_seg_metrics  # noqa: E402
from evaluation_3d import compute_3d_volume_metrics  # noqa: E402

from models_map import ModalityAdapter, UNetFeat, SegPipeline  # noqa: E402
from train_map import CLIENT_CONFIG, num_in_channels  # noqa: E402

torch.manual_seed(42)
np.random.seed(42)


def _parse_volume_slice(img_name):
    name = img_name[:-4] if img_name.endswith('.png') else img_name
    parts = name.split('_')
    try:
        slice_idx = int(parts[-1])
    except (ValueError, IndexError):
        slice_idx = 0
    vol_parts = parts[:-1]
    if vol_parts and vol_parts[-1] in ('flair', 't1ce', 't2', 't1', 'seg'):
        vol_parts = vol_parts[:-1]
    return '_'.join(vol_parts), slice_idx


@torch.no_grad()
def evaluate_client(client, args, device, log_file, csv_writer):
    in_ch = num_in_channels(client)
    unet = UNetFeat(in_channels=3, out_channels=1, base_filters=args.base_filters).to(device)
    adapter = ModalityAdapter(in_channels=in_ch).to(device)

    unet_path = os.path.join(args.model_dir, f"_personalized_seg_unet_client_{client}_best.pth")
    mod_path = os.path.join(args.model_dir, f"_personalized_modality_client_{client}_best.pth")
    unet.load_state_dict(torch.load(unet_path, map_location=device)['model_state_dict'])
    adapter.load_state_dict(torch.load(mod_path, map_location=device)['model_state_dict'])
    pipe = SegPipeline(adapter, unet).to(device)
    pipe.eval()
    print(f"  Loaded UNet  : {unet_path}")
    print(f"  Loaded modal : {mod_path}")

    df = pd.read_csv(args.csv_path)
    test_df = df[(df['client_id'] == client) & (df['split'] == 'test')].reset_index(drop=True)
    if len(test_df) == 0:
        print(f"  Warning: no test data for client {client}")
        return None

    dataset = SegValDataset(test_df, args.img_size, CLIENT_CONFIG, client_id=client)
    loader = DataLoader(dataset, batch_size=1, shuffle=False,
                        num_workers=args.num_workers, pin_memory=True)

    metrics = {'Dice': [], 'IoU': [], 'HD95': []}
    volume_data = {}

    for idx, (img, gt, label, blank) in enumerate(tqdm(loader, desc=f"  client {client}")):
        img = img.to(device, non_blocking=True)
        logits, _ = pipe(img)
        pred = (torch.sigmoid(logits) > args.threshold).float()
        pred_np = pred.squeeze().cpu().numpy().astype(np.uint8)
        gt_np = gt.squeeze().cpu().numpy().astype(np.uint8)
        if int(blank.item()) == 1:
            pred_np = np.zeros_like(gt_np)

        result = compute_seg_metrics(gt_np, pred_np)
        for k in metrics:
            metrics[k].append(result[k])

        img_name = str(test_df.iloc[idx, 0]).split('/')[-1]
        csv_writer.writelines(f"{img_name}, {result['Dice']:.3f}, {result['IoU']:.3f}, {result['HD95']:.3f}\n")

        if args.eval_3d:
            vol_id, sidx = _parse_volume_slice(img_name)
            volume_data.setdefault(vol_id, []).append((sidx, gt_np, pred_np))

    avg = {k: float(np.mean(v)) for k, v in metrics.items()}
    log_file.writelines(f"\nClient {client} (2D, per-slice)\n")
    for k, v in avg.items():
        log_file.writelines(f"  {k}: {v:.4f}\n")
    print(f"  Client {client} 2D: " + "  ".join(f"{k}={v:.4f}" for k, v in avg.items()))

    if args.eval_3d:
        agg = {'Dice': [], 'IoU': [], 'HD95': []}
        for vol_id in sorted(volume_data.keys()):
            slices = sorted(volume_data[vol_id], key=lambda x: x[0])
            gt_vol = np.stack([s[1] for s in slices], axis=0)
            pred_vol = np.stack([s[2] for s in slices], axis=0)
            m = compute_3d_volume_metrics(gt_vol, pred_vol)
            for k in agg:
                agg[k].append(m[k])
        avg3d = {f"3D {k}": float(np.mean(agg[k])) if agg[k] else float('nan') for k in agg}
        log_file.writelines(f"Client {client} (3D, per-volume)\n")
        for k, v in avg3d.items():
            log_file.writelines(f"  {k}: {v:.4f}\n")
        print(f"  Client {client} 3D: " + "  ".join(f"{k}={v:.4f}" for k, v in avg3d.items()))
        avg.update(avg3d)

    return avg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--clients', type=str, required=True)
    ap.add_argument('--csv_path', type=str, required=True)
    ap.add_argument('--model_dir', type=str, required=True)
    ap.add_argument('--base_filters', type=int, default=32)
    ap.add_argument('--img_size', type=int, default=224)
    ap.add_argument('--num_workers', type=int, default=4)
    ap.add_argument('--threshold', type=float, default=0.5)
    ap.add_argument('--gpu', type=int, default=0)
    ap.add_argument('--eval_3d', action='store_true')
    ap.add_argument('--out_dir', type=str, default=None)
    args = ap.parse_args()

    args.device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    device = args.device
    s = args.clients.strip()
    if s.startswith('['):
        parsed = ast.literal_eval(s)
        clients = list(parsed) if isinstance(parsed, (list, tuple)) else [int(parsed)]
    else:
        clients = [int(c) for c in s.split(',') if c.strip()]

    out_dir = args.out_dir or os.path.join(args.model_dir, "test_results")
    os.makedirs(out_dir, exist_ok=True)
    log_path = os.path.join(out_dir, "test_map.log")
    log_file = open(log_path, "w+")
    log_file.writelines(str(datetime.now()) + "\n")
    log_file.writelines(f"model_dir: {args.model_dir}\nthreshold: {args.threshold}\n")

    all_results = {}
    for client in clients:
        print(f"\n{'='*60}\nEvaluating Client {client}\n{'='*60}")
        csv_path = os.path.join(out_dir, f"test_result_client_{client}.csv")
        with open(csv_path, "w+") as csv_writer:
            csv_writer.writelines("Img Name, Dice, IoU, HD95\n")
            res = evaluate_client(client, args, device, log_file, csv_writer)
        if res is not None:
            all_results[client] = res

    if all_results:
        print(f"\n{'='*60}\nAverage across clients\n{'='*60}")
        log_file.writelines("\nAverage across clients\n")
        for metric in list(all_results.values())[0].keys():
            vals = [r[metric] for r in all_results.values() if metric in r]
            avg = float(np.mean(vals))
            print(f"  {metric}: {avg:.4f}")
            log_file.writelines(f"  {metric}: {avg:.4f}\n")

    log_file.close()
    print(f"\nResults written to {out_dir}")


if __name__ == "__main__":
    main()
