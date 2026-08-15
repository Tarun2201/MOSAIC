import os
import sys
import copy
import time
import argparse
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

# Reuse the existing datasets (train=CAM pseudo, val=GT) unchanged.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataset_segmentation import SegTrainDataset, SegValDataset  # noqa: E402

from models_map import ModalityAdapter, UNetFeat, SegPipeline  # noqa: E402
from losses_map import (  # noqa: E402
    PrototypeBank, refine_target, weighted_dice_bce, prototype_align_loss,
    local_prototypes, dice_score,
)

torch.manual_seed(42)
np.random.seed(42)
import random  # noqa: E402
random.seed(42)
torch.backends.cudnn.deterministic = True


CLIENT_CONFIG = {
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
    },
}


def setup_tee_logging(log_dir="nohups", log_file=None):
    if log_file is None:
        os.makedirs(log_dir, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(log_dir, f"train_map_{ts}.log")
    else:
        d = os.path.dirname(log_file)
        if d:
            os.makedirs(d, exist_ok=True)
    handle = open(log_file, "a", buffering=1)

    class Tee:
        def __init__(self, *s):
            self.s = s

        def write(self, data):
            for x in self.s:
                x.write(data)
            self.flush()

        def flush(self):
            for x in self.s:
                x.flush()

    sys.stdout = Tee(sys.stdout, handle)
    sys.stderr = Tee(sys.stderr, handle)
    print(f"Logging to {log_file}")
    return log_file


def num_in_channels(client_id):
    seen, n = set(), 0
    for m in CLIENT_CONFIG['clients'][str(client_id)]:
        if m not in seen:
            seen.add(m)
            n += 1
    return n


def aggregate_unets(client_unets, weights, device):
    """Weighted FedAvg of the shared backbone (params + buffers)."""
    global_unet = copy.deepcopy(client_unets[0]).to(device)
    gsd = global_unet.state_dict()
    csds = [m.state_dict() for m in client_unets]
    for k in gsd.keys():
        if gsd[k].dtype.is_floating_point:
            acc = torch.zeros_like(gsd[k])
            for w, csd in zip(weights, csds):
                acc += w * csd[k].to(device)
            gsd[k] = acc
        else:
            gsd[k] = csds[0][k]
    global_unet.load_state_dict(gsd)
    return global_unet


@torch.no_grad()
def ema_update(teacher, student, decay):
    for tp, sp in zip(teacher.parameters(), student.parameters()):
        tp.data.mul_(decay).add_(sp.data, alpha=1 - decay)
    for tb, sb in zip(teacher.buffers(), student.buffers()):
        if tb.dtype.is_floating_point:
            tb.data.mul_(decay).add_(sb.data, alpha=1 - decay)
        else:
            tb.data.copy_(sb.data)


def alpha_schedule(rnd, warmup, ramp_len, alpha_max):
    if rnd < warmup:
        return 0.0
    return alpha_max * min(1.0, (rnd - warmup) / max(ramp_len, 1))


def train_one_client(args, client, adapter, teacher, bank, global_unet,
                     train_loader, alpha, use_model, device):
    """One round of local training. Returns dict with the personalized student
    unet, reliability, and accumulated local prototypes."""
    unet = copy.deepcopy(global_unet).to(device)
    student = SegPipeline(adapter, unet).to(device)
    student.train()
    teacher.eval()

    opt = optim.Adam(student.parameters(), lr=args.learning_rate, weight_decay=1e-5)

    # local prototype accumulators (for the shared backbone's feature space)
    fdim = unet.feat_dim
    fg_sum = torch.zeros(fdim, device=device)
    bg_sum = torch.zeros(fdim, device=device)
    fg_cnt = torch.zeros((), device=device)
    bg_cnt = torch.zeros((), device=device)

    tot_seg = tot_proto = tot_cons = 0.0
    weight_accum = 0.0
    n_batches = 0

    for _ in range(args.epochs):
        for img, y_cam, label, blank in train_loader:
            img = img.to(device, non_blocking=True)
            y_cam = y_cam.to(device, non_blocking=True)
            label = label.to(device, non_blocking=True)

            # ---- teacher forward (denoised prediction + prototype consensus) ----
            with torch.no_grad():
                t_logits, t_feat = teacher(img)
                p_teacher = torch.sigmoid(t_logits)
                if use_model and bank.is_ready():
                    p_proto = bank.segment(t_feat)
                else:
                    p_proto = p_teacher  # no consensus yet -> fall back to teacher

            soft, weight = refine_target(
                y_cam, label, p_teacher, p_proto, alpha, use_model,
                weight_floor=args.weight_floor)

            # ---- student forward + loss ----
            opt.zero_grad(set_to_none=True)
            logits, feat = student(img)
            seg_loss = weighted_dice_bce(logits, soft, weight)
            proto_loss = prototype_align_loss(feat, soft, weight, bank)
            # light teacher-student consistency (stabilises the moving target)
            cons_loss = ((torch.sigmoid(logits) - p_teacher) ** 2 * weight).mean()

            loss = seg_loss + args.proto_weight * proto_loss + args.cons_weight * cons_loss
            loss.backward()
            opt.step()

            ema_update(teacher, student, args.ema_decay)

            # ---- accumulate local prototypes from the STUDENT backbone ----
            with torch.no_grad():
                lp = local_prototypes(feat.detach(), (soft > 0.5).float(), weight)
                if lp is not None:
                    fv, fc, bv, bc = lp
                    fg_sum += fv * fc
                    bg_sum += bv * bc
                    fg_cnt += fc
                    bg_cnt += bc

            tot_seg += float(seg_loss)
            tot_proto += float(proto_loss)
            tot_cons += float(cons_loss)
            weight_accum += float(weight.mean())
            n_batches += 1

    n_batches = max(n_batches, 1)
    reliability = weight_accum / n_batches  # mean pixel agreement this round
    local_proto = None
    if float(fg_cnt) > 0 and float(bg_cnt) > 0:
        local_proto = {
            'fg': (fg_sum / fg_cnt).detach(), 'fg_cnt': float(fg_cnt),
            'bg': (bg_sum / bg_cnt).detach(), 'bg_cnt': float(bg_cnt),
        }
    return {
        'unet': unet,
        'reliability': reliability,
        'local_proto': local_proto,
        'seg': tot_seg / n_batches,
        'proto': tot_proto / n_batches,
        'cons': tot_cons / n_batches,
    }


@torch.no_grad()
def validate(adapter, unet, val_loader, device):
    pipe = SegPipeline(adapter, unet).to(device)
    pipe.eval()
    dices = []
    n = 0
    for img, gt, label, blank in val_loader:
        img = img.to(device, non_blocking=True)
        gt = gt.to(device, non_blocking=True)
        logits, _ = pipe(img)
        if blank.any():
            bmask = blank.to(device).view(-1, 1, 1, 1).bool()
            logits = torch.where(bmask, torch.full_like(logits, -1e4), logits)
        bs = img.size(0)
        dices.append(dice_score(logits, gt) * bs)
        n += bs
    return float(np.sum(dices) / max(n, 1))


def aggregate_prototypes(local_protos, device):
    """Count-weighted mean fg/bg prototype across clients (modality-invariant)."""
    valid = [lp for lp in local_protos if lp is not None]
    if not valid:
        return None
    fg = torch.zeros_like(valid[0]['fg'])
    bg = torch.zeros_like(valid[0]['bg'])
    fw = bw = 0.0
    for lp in valid:
        fg += lp['fg'] * lp['fg_cnt']
        bg += lp['bg'] * lp['bg_cnt']
        fw += lp['fg_cnt']
        bw += lp['bg_cnt']
    return fg / (fw + 1e-6), bg / (bw + 1e-6)


def federated_train(args):
    device = args.device
    print(f"FedMAP-R | clients={args.clients} | device={device}")
    t0 = time.time()

    temp = ""
    for c in args.clients:
        temp = f"{c}_" + temp
    save_dir = os.path.join(args.save_dir, temp, "map_refine") + "/"
    os.makedirs(save_dir, exist_ok=True)
    print(f"Saving models to {save_dir}")

    manifest = pd.read_csv(args.manifest_path)
    split_df = pd.read_csv(args.csv_path)

    global_unet = UNetFeat(in_channels=3, out_channels=1, base_filters=args.base_filters).to(device)
    fdim = global_unet.feat_dim

    adapters, teachers, banks = {}, {}, {}
    train_loaders, val_loaders, num_train = {}, {}, []

    lk = {'num_workers': args.num_workers, 'pin_memory': True}
    if args.num_workers > 0:
        lk['persistent_workers'] = True
        lk['prefetch_factor'] = 2

    for client in args.clients:
        in_ch = num_in_channels(client)
        adapters[client] = ModalityAdapter(in_channels=in_ch).to(device)
        # teacher = frozen EMA copy of (adapter + a fresh backbone)
        teachers[client] = SegPipeline(
            copy.deepcopy(adapters[client]),
            copy.deepcopy(global_unet)).to(device)
        for p in teachers[client].parameters():
            p.requires_grad_(False)
        banks[client] = PrototypeBank(fdim, tau=args.proto_tau).to(device)

        m_train = manifest[(manifest['client_id'] == client) & (manifest['split'] == 'train')]
        train_ds = SegTrainDataset(m_train, args.img_size, CLIENT_CONFIG, client_id=client)
        num_train.append(len(train_ds))
        train_loaders[client] = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, **lk)

        v_df = split_df[(split_df['client_id'] == client) & (split_df['split'] == 'val')]
        val_ds = SegValDataset(v_df, args.img_size, CLIENT_CONFIG, client_id=client)
        val_loaders[client] = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, **lk)
        print(f"  client {client}: in_ch={in_ch} train(pseudo)={len(train_ds)} val(gt)={len(val_ds)}")

    size_w = np.array(num_train, dtype=np.float64)
    best_dice = {c: 0.0 for c in args.clients}
    best_avg = 0.0
    patience = 0

    for rnd in range(args.num_rounds):
        if rnd % 5 == 0:
            torch.cuda.empty_cache()
        alpha = alpha_schedule(rnd, args.warmup_rounds, args.ramp_rounds, args.alpha_max)
        use_model = rnd >= args.warmup_rounds
        print(f"\n{'='*60}\nRound {rnd+1}/{args.num_rounds}  alpha={alpha:.2f}  use_model={use_model}\n{'='*60}")

        client_unets, reliabilities, local_protos, dices = [], [], [], []
        for i, client in enumerate(args.clients):
            res = train_one_client(
                args, client, adapters[client], teachers[client], banks[client],
                global_unet, train_loaders[client], alpha, use_model, device)
            client_unets.append(res['unet'])
            reliabilities.append(res['reliability'])
            local_protos.append(res['local_proto'])

            vdice = validate(adapters[client], res['unet'], val_loaders[client], device)
            dices.append(vdice)
            print(f"  client {client}: seg={res['seg']:.4f} proto={res['proto']:.4f} "
                  f"cons={res['cons']:.4f} reliab={res['reliability']:.3f} val_dice(GT)={vdice:.4f}")

            if vdice > best_dice[client]:
                best_dice[client] = vdice
                torch.save({'round': rnd, 'model_state_dict': res['unet'].state_dict(), 'val_dice': vdice},
                           os.path.join(save_dir, f"_personalized_seg_unet_client_{client}_best.pth"))
                torch.save({'round': rnd, 'model_state_dict': adapters[client].state_dict(), 'val_dice': vdice},
                           os.path.join(save_dir, f"_personalized_modality_client_{client}_best.pth"))
                print(f"    -> new best for client {client} (dice {vdice:.4f}), saved")

        avg_dice = float(np.mean(dices))
        print(f"\nAverage val Dice(GT): {avg_dice:.4f}")

        # ---- reliability-weighted FedAvg of the shared backbone ----
        rel = np.array(reliabilities, dtype=np.float64)
        rel = np.clip(rel, 1e-3, None)
        agg_w = size_w * rel
        agg_w = agg_w / agg_w.sum()
        print("  FedAvg weights (size x reliability): " +
              ", ".join(f"{c}:{w:.3f}" for c, w in zip(args.clients, agg_w)))
        global_unet = aggregate_unets(client_unets, agg_w, device)

        # ---- aggregate modality-invariant prototypes, broadcast ----
        agg_proto = aggregate_prototypes(local_protos, device)
        if agg_proto is not None:
            gfg, gbg = agg_proto
            for client in args.clients:
                banks[client].set_global(gfg, gbg)
            print("  Updated global fg/bg prototypes (modality-invariant)")

        # ---- early stopping on avg GT val Dice ----
        if avg_dice > best_avg + 1e-4:
            best_avg = avg_dice
            patience = 0
            print("  New best average val Dice. Counter reset.")
        else:
            patience += 1
            print(f"  No improvement. Patience {patience}/{args.patience}")
            if patience >= args.patience:
                print(f"\nEarly stopping after {args.patience} rounds without improvement.")
                break

        if (rnd + 1) % 10 == 0:
            torch.save({'round': rnd, 'model_state_dict': global_unet.state_dict()},
                       os.path.join(save_dir, f"_global_seg_unet_round_{rnd}.pth"))

    print("\nTraining completed!")
    for c in args.clients:
        print(f"  client {c}: best val_dice(GT)={best_dice[c]:.4f}")
    print(f"  mean best val_dice(GT)={np.mean([best_dice[c] for c in args.clients]):.4f}")
    print(f"Total time: {(time.time()-t0)/60:.2f} min")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument('--clients', type=str, default="1,2,3,4")
    ap.add_argument('--num_rounds', type=int, default=100)
    ap.add_argument('--epochs', type=int, default=1)
    ap.add_argument('--batch_size', type=int, default=16)
    ap.add_argument('--img_size', type=int, default=224)
    ap.add_argument('--num_workers', type=int, default=4)
    ap.add_argument('--learning_rate', type=float, default=1e-3)
    ap.add_argument('--patience', type=int, default=25)

    ap.add_argument('--csv_path', type=str, required=True)
    ap.add_argument('--manifest_path', type=str, required=True)
    ap.add_argument('--save_dir', type=str, default='./results/seg_map_refine/')
    ap.add_argument('--device', type=str, default='cuda:0')
    ap.add_argument('--base_filters', type=int, default=32)

    # FedMAP-R hyper-parameters
    ap.add_argument('--warmup_rounds', type=int, default=5,
                    help='Rounds trained on the raw CAM before refinement kicks in.')
    ap.add_argument('--ramp_rounds', type=int, default=20,
                    help='Rounds over which trust in the model consensus ramps to alpha_max.')
    ap.add_argument('--alpha_max', type=float, default=0.7,
                    help='Max fraction of the target coming from the model consensus.')
    ap.add_argument('--ema_decay', type=float, default=0.99)
    ap.add_argument('--proto_tau', type=float, default=0.1)
    ap.add_argument('--proto_weight', type=float, default=0.1)
    ap.add_argument('--cons_weight', type=float, default=0.05)
    ap.add_argument('--weight_floor', type=float, default=0.1,
                    help='Minimum per-pixel loss weight (never fully ignore a pixel).')

    ap.add_argument('--log_dir', type=str, default='nohups')
    ap.add_argument('--log_file', type=str, default=None)
    args = ap.parse_args()

    setup_tee_logging(args.log_dir, args.log_file)
    args.clients = [int(c) for c in args.clients.split(',')]
    if str(args.device).startswith('cuda') and not torch.cuda.is_available():
        args.device = 'cpu'
    federated_train(args)
