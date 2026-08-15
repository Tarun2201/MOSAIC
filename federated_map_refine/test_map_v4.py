import os
import sys
import argparse
from datetime import datetime

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from evaluation import compute_seg_metrics  # noqa: E402
from evaluation_3d import compute_3d_volume_metrics  # noqa: E402
from dataset_25d import SegValDataset25D  # noqa: E402
from models_map_v4 import ModalityAdapter, SegPipeline25D, build_v4_unet  # noqa: E402
from train_map import CLIENT_CONFIG, num_in_channels  # noqa: E402
from test_map import _parse_volume_slice  # noqa: E402

torch.manual_seed(42); np.random.seed(42)


@torch.no_grad()
def evaluate_client(client, args, device, log_file, csv_writer):
    ckpt = torch.load(os.path.join(args.model_dir, f"_personalized_seg_unet_client_{client}_best.pth"),
                      map_location=device)
    K = ckpt.get('K', args.K)
    unet = build_v4_unet(args.base_filters, K, device)
    unet.load_state_dict(ckpt['model_state_dict'])
    ad = ModalityAdapter(num_in_channels(client)).to(device)
    ad.load_state_dict(torch.load(os.path.join(args.model_dir, f"_personalized_modality_client_{client}_best.pth"),
                                  map_location=device)['model_state_dict'])
    pipe = SegPipeline25D(ad, unet, K).to(device).eval()
    print(f"  client {client}: K={K}")

    df = pd.read_csv(args.csv_path)
    test_df = df[(df['client_id'] == client) & (df['split'] == 'test')].reset_index(drop=True)
    if len(test_df) == 0:
        return None
    loader = DataLoader(SegValDataset25D(test_df, args.img_size, CLIENT_CONFIG, client_id=client, context=K // 2),
                        batch_size=1, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    metrics = {'Dice': [], 'IoU': [], 'HD95': []}
    volume_data = {}
    for idx, (stack, gt, label, blank) in enumerate(tqdm(loader, desc=f"  client {client}")):
        stack = stack.to(device)
        logits, _, cls_logit = pipe(stack)
        prob = torch.sigmoid(logits); cls = cls_logit
        if args.tta:
            lf, _, cf = pipe(torch.flip(stack, dims=[4]))
            prob = 0.5 * (prob + torch.flip(torch.sigmoid(lf), dims=[3]))
            cls = 0.5 * (cls + cf)
        present = float(torch.sigmoid(cls).view(-1)[0]) >= args.gate_tau
        pred = (prob > 0.5).float().squeeze().cpu().numpy().astype(np.uint8)
        gt_np = gt.squeeze().cpu().numpy().astype(np.uint8)
        if int(blank.item()) == 1 or not present:
            pred = np.zeros_like(gt_np)
        r = compute_seg_metrics(gt_np, pred)
        for k in metrics:
            metrics[k].append(r[k])
        name = str(test_df.iloc[idx, 0]).split('/')[-1]
        csv_writer.writelines(f"{name}, {r['Dice']:.3f}, {r['IoU']:.3f}, {r['HD95']:.3f}\n")
        if args.eval_3d:
            vid, sidx = _parse_volume_slice(name)
            volume_data.setdefault(vid, []).append((sidx, gt_np, pred))

    avg = {k: float(np.mean(v)) for k, v in metrics.items()}
    print(f"  Client {client} 2D: " + "  ".join(f"{k}={v:.4f}" for k, v in avg.items()))
    log_file.writelines(f"Client {client} 2D: " + "  ".join(f"{k}={v:.4f}" for k, v in avg.items()) + "\n")
    if args.eval_3d:
        agg = {'Dice': [], 'IoU': [], 'HD95': []}
        for vid in sorted(volume_data.keys()):
            sl = sorted(volume_data[vid], key=lambda x: x[0])
            gv = np.stack([s[1] for s in sl], 0); pv = np.stack([s[2] for s in sl], 0)
            m = compute_3d_volume_metrics(gv, pv)
            for k in agg:
                agg[k].append(m[k])
        avg3d = {f"3D {k}": float(np.mean(agg[k])) if agg[k] else float('nan') for k in agg}
        print(f"  Client {client} 3D: " + "  ".join(f"{k}={v:.4f}" for k, v in avg3d.items()))
        log_file.writelines(f"Client {client} 3D: " + "  ".join(f"{k}={v:.4f}" for k, v in avg3d.items()) + "\n")
        avg.update(avg3d)
    return avg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--clients', type=str, required=True)
    ap.add_argument('--csv_path', type=str, required=True)
    ap.add_argument('--model_dir', type=str, required=True)
    ap.add_argument('--base_filters', type=int, default=32)
    ap.add_argument('--img_size', type=int, default=224)
    ap.add_argument('--num_workers', type=int, default=6)
    ap.add_argument('--K', type=int, default=3)
    ap.add_argument('--gate_tau', type=float, default=0.5)
    ap.add_argument('--tta', action='store_true')
    ap.add_argument('--gpu', type=int, default=0)
    ap.add_argument('--eval_3d', action='store_true')
    ap.add_argument('--out_dir', type=str, default=None)
    args = ap.parse_args()
    args.device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    device = args.device
    clients = [int(c) for c in args.clients.split(',')]
    out_dir = args.out_dir or os.path.join(args.model_dir, "test_results")
    os.makedirs(out_dir, exist_ok=True)
    log_file = open(os.path.join(out_dir, "test_map_v4.log"), "w+")
    log_file.writelines(str(datetime.now()) + f"\ntta={args.tta} gate_tau={args.gate_tau}\n")
    all_results = {}
    for c in clients:
        print(f"\n{'='*50}\nClient {c}\n{'='*50}")
        with open(os.path.join(out_dir, f"test_result_client_{c}.csv"), "w+") as w:
            w.writelines("Img Name, Dice, IoU, HD95\n")
            r = evaluate_client(c, args, device, log_file, w)
        if r:
            all_results[c] = r
    if all_results:
        print(f"\n{'='*50}\nAverage across clients\n{'='*50}")
        log_file.writelines("\nAverage across clients\n")
        for metric in list(all_results.values())[0].keys():
            v = float(np.mean([r[metric] for r in all_results.values() if metric in r]))
            print(f"  {metric}: {v:.4f}"); log_file.writelines(f"  {metric}: {v:.4f}\n")
    log_file.close()
    print(f"\nResults -> {out_dir}")


if __name__ == "__main__":
    main()
