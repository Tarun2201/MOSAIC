import os
import sys
import copy
import time
import argparse

import numpy as np
import pandas as pd
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataset_25d import SegTrainDataset25D, SegValDataset25D  # noqa: E402
from models_map_v4 import ModalityAdapter, SegPipeline25D, build_v4_unet, warm_start_25d  # noqa: E402
from losses_map_v2 import (  # noqa: E402
    PrototypeBank, prototype_align_loss, local_prototypes, dice_score,
    arbitrate_target, focal_tversky, gated_crf_loss,
)
from train_map import (  # noqa: E402
    CLIENT_CONFIG, num_in_channels, setup_tee_logging, aggregate_unets,
    ema_update, alpha_schedule, aggregate_prototypes,
)

torch.manual_seed(42)
np.random.seed(42)
import random  # noqa: E402
random.seed(42)
torch.backends.cudnn.deterministic = True


def train_one_client(args, client, adapter, teacher, bank, global_unet,
                     train_loader, alpha, use_model, device):
    unet = copy.deepcopy(global_unet).to(device)
    student = SegPipeline25D(adapter, unet, args.K).to(device)
    student.train(); teacher.eval()
    opt = optim.Adam(student.parameters(), lr=args.learning_rate, weight_decay=1e-5)

    fdim = unet.feat_dim
    fg_sum = torch.zeros(fdim, device=device); bg_sum = torch.zeros(fdim, device=device)
    fg_cnt = torch.zeros((), device=device); bg_cnt = torch.zeros((), device=device)
    tot = {'seg': 0.0, 'proto': 0.0, 'cons': 0.0, 'cls': 0.0, 'crf': 0.0}
    weight_accum = 0.0; n_batches = 0
    kc = args.K // 2

    for _ in range(args.epochs):
        for stack, y_cam, label, blank in train_loader:
            stack = stack.to(device, non_blocking=True)
            y_cam = y_cam.to(device, non_blocking=True)
            label = label.to(device, non_blocking=True)
            center_img = stack[:, kc]                      # (B, num_mod, H, W) for CRF

            with torch.no_grad():
                t_logits, t_feat, _ = teacher(stack)
                p_teacher = torch.sigmoid(t_logits)
                p_proto = bank.segment(t_feat) if (use_model and bank.is_ready()) else p_teacher

            soft, weight = arbitrate_target(y_cam, label, p_teacher, p_proto, alpha,
                                            use_model, beta=args.vote_beta,
                                            weight_floor=args.weight_floor)

            opt.zero_grad(set_to_none=True)
            logits, feat, cls_logit = student(stack)
            p_student = torch.sigmoid(logits)
            seg_loss = focal_tversky(logits, soft, weight, alpha=args.tv_alpha,
                                     beta=args.tv_beta, gamma=args.tv_gamma)
            proto_loss = prototype_align_loss(feat, soft, weight, bank)
            cons_loss = ((p_student - p_teacher) ** 2 * weight).mean()
            cls_loss = F.binary_cross_entropy_with_logits(cls_logit.view(-1), label.float())
            crf_loss = gated_crf_loss(p_student, center_img, kernel=args.crf_kernel,
                                      sigma_int=args.crf_sigma_int, sigma_xy=args.crf_sigma_xy) \
                if args.crf_weight > 0 else logits.new_zeros(())
            loss = (seg_loss + args.proto_weight * proto_loss + args.cons_weight * cons_loss
                    + args.cls_weight * cls_loss + args.crf_weight * crf_loss)
            loss.backward(); opt.step()
            ema_update(teacher, student, args.ema_decay)

            with torch.no_grad():
                lp = local_prototypes(feat.detach(), (soft > 0.5).float(), weight)
                if lp is not None:
                    fv, fc, bv, bc = lp
                    fg_sum += fv * fc; bg_sum += bv * bc; fg_cnt += fc; bg_cnt += bc
            tot['seg'] += float(seg_loss); tot['proto'] += float(proto_loss)
            tot['cons'] += float(cons_loss); tot['cls'] += float(cls_loss); tot['crf'] += float(crf_loss)
            weight_accum += float(weight.mean()); n_batches += 1

    n_batches = max(n_batches, 1)
    local_proto = None
    if float(fg_cnt) > 0 and float(bg_cnt) > 0:
        local_proto = {'fg': (fg_sum / fg_cnt).detach(), 'fg_cnt': float(fg_cnt),
                       'bg': (bg_sum / bg_cnt).detach(), 'bg_cnt': float(bg_cnt)}
    return {'unet': unet, 'reliability': weight_accum / n_batches, 'local_proto': local_proto,
            **{k: v / n_batches for k, v in tot.items()}}


@torch.no_grad()
def validate(adapter, unet, val_loader, device, gate_tau, K, tta=False):
    pipe = SegPipeline25D(adapter, unet, K).to(device); pipe.eval()
    dices, n = [], 0
    for stack, gt, label, blank in val_loader:
        stack = stack.to(device, non_blocking=True); gt = gt.to(device, non_blocking=True)
        logits, _, cls_logit = pipe(stack)
        prob = torch.sigmoid(logits); cls = cls_logit
        if tta:
            lf, _, cf = pipe(torch.flip(stack, dims=[4]))
            prob = 0.5 * (prob + torch.flip(torch.sigmoid(lf), dims=[3]))
            cls = 0.5 * (cls + cf)
        present = (torch.sigmoid(cls).view(-1, 1, 1, 1) >= gate_tau)
        logit_eq = torch.log(prob.clamp(1e-6, 1 - 1e-6) / (1 - prob.clamp(1e-6, 1 - 1e-6)))
        logit_eq = torch.where(present, logit_eq, torch.full_like(logit_eq, -1e4))
        if blank.any():
            bmask = blank.to(device).view(-1, 1, 1, 1).bool()
            logit_eq = torch.where(bmask, torch.full_like(logit_eq, -1e4), logit_eq)
        bs = stack.size(0); dices.append(dice_score(logit_eq, gt) * bs); n += bs
    return float(np.sum(dices) / max(n, 1))


def federated_train(args):
    device = args.device
    print(f"FedMAP-R v4 (2.5D, K={args.K}) | clients={args.clients} | device={device}")
    t0 = time.time()
    temp = ""
    for c in args.clients:
        temp = f"{c}_" + temp
    save_dir = os.path.join(args.save_dir, temp, "map_refine_v4") + "/"
    os.makedirs(save_dir, exist_ok=True)
    print(f"Saving models to {save_dir}")

    manifest = pd.read_csv(args.manifest_path)
    split_df = pd.read_csv(args.csv_path)
    context = args.K // 2

    global_unet = build_v4_unet(args.base_filters, args.K, device)
    fdim = global_unet.feat_dim
    adapters, teachers, banks = {}, {}, {}
    train_loaders, val_loaders, num_train = {}, {}, []
    lk = {'num_workers': args.num_workers, 'pin_memory': True}
    if args.num_workers > 0:
        lk['persistent_workers'] = True; lk['prefetch_factor'] = 2

    warm = False
    for client in args.clients:
        in_ch = num_in_channels(client)
        adapters[client] = ModalityAdapter(in_channels=in_ch).to(device)
        if args.init_from_v2:
            a_path = os.path.join(args.init_from_v2, f"_personalized_modality_client_{client}_best.pth")
            u_path = os.path.join(args.init_from_v2, f"_personalized_seg_unet_client_{client}_best.pth")
            if os.path.exists(a_path):
                adapters[client].load_state_dict(torch.load(a_path, map_location=device)['model_state_dict'])
            if os.path.exists(u_path):
                v2sd = torch.load(u_path, map_location=device)['model_state_dict']
                tmp = build_v4_unet(args.base_filters, args.K, device)
                warm_start_25d(tmp, v2sd, args.K, verbose=(client == args.clients[0]))
                warm = True
                if client == args.clients[0]:
                    global_unet.load_state_dict(tmp.state_dict())
        teachers[client] = SegPipeline25D(copy.deepcopy(adapters[client]),
                                          copy.deepcopy(global_unet), args.K).to(device)
        for p in teachers[client].parameters():
            p.requires_grad_(False)
        banks[client] = PrototypeBank(fdim, tau=args.proto_tau).to(device)

        m_train = manifest[(manifest['client_id'] == client) & (manifest['split'] == 'train')]
        tr = SegTrainDataset25D(m_train, args.img_size, CLIENT_CONFIG, client_id=client, context=context)
        num_train.append(len(tr))
        train_loaders[client] = DataLoader(tr, batch_size=args.batch_size, shuffle=True, **lk)
        v_df = split_df[(split_df['client_id'] == client) & (split_df['split'] == 'val')]
        va = SegValDataset25D(v_df, args.img_size, CLIENT_CONFIG, client_id=client, context=context)
        val_loaders[client] = DataLoader(va, batch_size=args.batch_size, shuffle=False, **lk)
        print(f"  client {client}: in_ch={in_ch} train={len(tr)} val(GT)={len(va)}")
    print(f"Warm-start from v2: {'YES' if warm else 'NO'}")

    size_w = np.array(num_train, dtype=np.float64)
    best_dice = {c: 0.0 for c in args.clients}
    best_avg = 0.0; patience = 0

    for rnd in range(args.num_rounds):
        if rnd % 5 == 0:
            torch.cuda.empty_cache()
        alpha = alpha_schedule(rnd, args.warmup_rounds, args.ramp_rounds, args.alpha_max)
        use_model = rnd >= args.warmup_rounds
        print(f"\n{'='*60}\nRound {rnd+1}/{args.num_rounds}  alpha={alpha:.2f}\n{'='*60}")
        client_unets, reliabilities, local_protos, dices = [], [], [], []
        for client in args.clients:
            res = train_one_client(args, client, adapters[client], teachers[client],
                                   banks[client], global_unet, train_loaders[client],
                                   alpha, use_model, device)
            client_unets.append(res['unet']); reliabilities.append(res['reliability'])
            local_protos.append(res['local_proto'])
            vdice = validate(adapters[client], res['unet'], val_loaders[client], device,
                             args.gate_tau, args.K, tta=args.tta)
            dices.append(vdice)
            print(f"  client {client}: seg={res['seg']:.3f} crf={res['crf']:.3f} cls={res['cls']:.3f} "
                  f"reliab={res['reliability']:.3f} val_dice(GT)={vdice:.4f}")
            if vdice > best_dice[client]:
                best_dice[client] = vdice
                torch.save({'round': rnd, 'model_state_dict': res['unet'].state_dict(), 'val_dice': vdice, 'K': args.K},
                           os.path.join(save_dir, f"_personalized_seg_unet_client_{client}_best.pth"))
                torch.save({'round': rnd, 'model_state_dict': adapters[client].state_dict(), 'val_dice': vdice},
                           os.path.join(save_dir, f"_personalized_modality_client_{client}_best.pth"))
                print(f"    -> new best client {client} ({vdice:.4f}), saved")

        avg_dice = float(np.mean(dices))
        print(f"\nAverage val Dice(GT): {avg_dice:.4f}")
        rel = np.clip(np.array(reliabilities, dtype=np.float64), 1e-3, None)
        agg_w = size_w * rel; agg_w = agg_w / agg_w.sum()
        global_unet = aggregate_unets(client_unets, agg_w, device)
        agg_proto = aggregate_prototypes(local_protos, device)
        if agg_proto is not None:
            gfg, gbg = agg_proto
            for client in args.clients:
                banks[client].set_global(gfg, gbg)
        if avg_dice > best_avg + 1e-4:
            best_avg = avg_dice; patience = 0; print("  New best average. Reset.")
        else:
            patience += 1; print(f"  No improvement. Patience {patience}/{args.patience}")
            if patience >= args.patience:
                print(f"\nEarly stopping."); break

    print("\nTraining completed!")
    for c in args.clients:
        print(f"  client {c}: best val_dice(GT)={best_dice[c]:.4f}")
    print(f"  mean best val_dice(GT)={np.mean([best_dice[c] for c in args.clients]):.4f}")
    print(f"Total time: {(time.time()-t0)/60:.2f} min")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument('--clients', type=str, default="1,2,3,4")
    ap.add_argument('--num_rounds', type=int, default=40)
    ap.add_argument('--epochs', type=int, default=1)
    ap.add_argument('--batch_size', type=int, default=16)
    ap.add_argument('--img_size', type=int, default=224)
    ap.add_argument('--num_workers', type=int, default=6)
    ap.add_argument('--learning_rate', type=float, default=5e-4)
    ap.add_argument('--patience', type=int, default=12)
    ap.add_argument('--csv_path', type=str, required=True)
    ap.add_argument('--manifest_path', type=str, required=True)
    ap.add_argument('--save_dir', type=str, default='./results/seg_map_refine/')
    ap.add_argument('--device', type=str, default='cuda:0')
    ap.add_argument('--base_filters', type=int, default=32)
    ap.add_argument('--K', type=int, default=3, help='context slices (odd)')
    ap.add_argument('--init_from_v2', type=str, default='')
    ap.add_argument('--tta', action='store_true', help='flip TTA at val/test')

    ap.add_argument('--warmup_rounds', type=int, default=2)
    ap.add_argument('--ramp_rounds', type=int, default=8)
    ap.add_argument('--alpha_max', type=float, default=0.8)
    ap.add_argument('--ema_decay', type=float, default=0.99)
    ap.add_argument('--vote_beta', type=float, default=6.0)
    ap.add_argument('--weight_floor', type=float, default=0.1)
    ap.add_argument('--proto_tau', type=float, default=0.1)
    ap.add_argument('--tv_alpha', type=float, default=0.3)
    ap.add_argument('--tv_beta', type=float, default=0.7)
    ap.add_argument('--tv_gamma', type=float, default=1.0)
    ap.add_argument('--proto_weight', type=float, default=0.1)
    ap.add_argument('--cons_weight', type=float, default=0.05)
    ap.add_argument('--cls_weight', type=float, default=0.5)
    ap.add_argument('--crf_weight', type=float, default=0.1)
    ap.add_argument('--crf_kernel', type=int, default=5)
    ap.add_argument('--crf_sigma_int', type=float, default=0.15)
    ap.add_argument('--crf_sigma_xy', type=float, default=6.0)
    ap.add_argument('--gate_tau', type=float, default=0.5)
    ap.add_argument('--log_dir', type=str, default='nohups')
    ap.add_argument('--log_file', type=str, default=None)
    args = ap.parse_args()

    setup_tee_logging(args.log_dir, args.log_file)
    args.clients = [int(c) for c in args.clients.split(',')]
    if str(args.device).startswith('cuda') and not torch.cuda.is_available():
        args.device = 'cpu'
    federated_train(args)
