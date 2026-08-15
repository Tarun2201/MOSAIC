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
from dataset_25d import SegTrainDataset25D, SegTrainDataset25DTrimap, SegValDataset25D  # noqa: E402
from models_map_v4 import ModalityAdapter, SegPipeline25D, build_v4_unet, warm_start_25d  # noqa: E402
from losses_map_v2 import (  # noqa: E402
    PrototypeBank, prototype_align_loss, local_prototypes, dice_score,
    arbitrate_target, focal_tversky, gated_crf_loss,
)
from losses_v6 import (  # noqa: E402
    multiscale_gated_crf_loss, healthy_feat_pool, healthy_distill_loss,
)
from losses_spa import SpectralAlign, RegionConditionedSpectralAlign, aggregate_freq_stats  # noqa: E402
from train_map import (  # noqa: E402
    CLIENT_CONFIG, num_in_channels, setup_tee_logging, aggregate_unets,
    ema_update, alpha_schedule, aggregate_prototypes,
)
from train_map_v4 import validate  # noqa: E402  (2.5D validate + TTA)

torch.manual_seed(42)
np.random.seed(42)
import random  # noqa: E402
random.seed(42)
torch.backends.cudnn.deterministic = True


def _avg_stats(lst):
    lst = [s for s in lst if s]
    if not lst:
        return None
    keys = set().union(*[s.keys() for s in lst])
    out = {}
    for k in keys:
        vals = [s[k] for s in lst if k in s]
        if vals:
            out[k] = torch.stack(vals).mean(0)
    return out


@torch.no_grad()
def init_spectral_prototypes(args, adapters, spas, rcsas, train_loaders, agg_w, device, n_batches=4):
    """Round 0: seed the global spectral prototypes with a cross-client consensus
    BEFORE the training loop, so the SPA / RCSA alignment is active (nonzero) and
    meaningful from round 1 (otherwise the is_ready() guard makes round 1 exactly
    0, since the consensus is only formed at the first server aggregation)."""
    kc = args.K // 2
    per_spa, per_rcsa = [], []
    for client in args.clients:
        adapters[client].eval()
        ss, rr = [], []
        for bi, (stack, y_cam, wmap, label, blank) in enumerate(train_loaders[client]):
            stack = stack.to(device, non_blocking=True)
            y_cam = y_cam.to(device, non_blocking=True)
            ac = adapters[client](stack[:, kc])
            ss.append(spas[client].compute_local_statistics(ac))
            rr.append(rcsas[client].compute_local_statistics(ac, y_cam))
            if bi + 1 >= n_batches:
                break
        adapters[client].train()
        per_spa.append(_avg_stats(ss))
        per_rcsa.append(_avg_stats(rr))
    spa_g = aggregate_freq_stats(per_spa, agg_w)
    rcsa_g = aggregate_freq_stats(per_rcsa, agg_w)
    for client in args.clients:
        if spa_g:
            spas[client].update_global_statistics(spa_g)
        if rcsa_g:
            rcsas[client].update_global_statistics(rcsa_g)
    return spa_g is not None, rcsa_g is not None


def train_one_client(args, client, adapter, teacher, bank, 
                     spa, rcsa, align_scale,
                     global_unet, train_loader, alpha, use_model, mu_healthy, fnfd_w, device):
    unet = copy.deepcopy(global_unet).to(device)
    student = SegPipeline25D(adapter, unet, args.K).to(device)
    student.train(); teacher.eval()
    opt = optim.Adam(student.parameters(), lr=args.learning_rate, weight_decay=1e-5)

    fdim = unet.feat_dim
    fg_sum = torch.zeros(fdim, device=device); bg_sum = torch.zeros(fdim, device=device)
    fg_cnt = torch.zeros((), device=device); bg_cnt = torch.zeros((), device=device)
    h_sum = torch.zeros(fdim, device=device); h_cnt = 0.0     # healthy-feature accumulator
    spa_stats, rcsa_stats = [], []
    tot = {'seg': 0.0, 'proto': 0.0, 'cls': 0.0, 'crf': 0.0, 'spa': 0.0, 'rcsa': 0.0, 'fnfd': 0.0}
    weight_accum = 0.0; n_batches = 0; nan_batches = 0
    kc = args.K // 2

    for _ in range(args.epochs):
        for stack, y_cam, wmap, label, blank in train_loader:
            stack = stack.to(device, non_blocking=True)
            y_cam = y_cam.to(device, non_blocking=True)
            wmap = wmap.to(device, non_blocking=True)
            label = label.to(device, non_blocking=True)
            center_img = stack[:, kc]

            with torch.no_grad():
                t_logits, t_feat, _ = teacher(stack)
                p_teacher = torch.sigmoid(t_logits)
                p_proto = bank.segment(t_feat) if (use_model and bank.is_ready()) else p_teacher
            soft, weight = arbitrate_target(y_cam, label, p_teacher, p_proto, alpha,
                                            use_model, beta=args.vote_beta,
                                            weight_floor=args.weight_floor)
            # Zero the loss weight on IGNORE pixels (noisy boundary + empty-CAM
            # positive slices). focal_tversky/proto/cons all multiply by `weight`.
            weight = weight * wmap

            opt.zero_grad(set_to_none=True)
            logits, feat, cls_logit = student(stack)
            p_student = torch.sigmoid(logits)
            ad_center = student.adapter(center_img)         # (B,3,H,W) modality-aligned features

            seg_loss = focal_tversky(logits, soft, weight, alpha=args.tv_alpha,
                                     beta=args.tv_beta, gamma=args.tv_gamma)
            proto_loss = prototype_align_loss(feat, soft, weight, bank)
            cons_loss = ((p_student - p_teacher) ** 2 * weight).mean()
            cls_loss = F.binary_cross_entropy_with_logits(cls_logit.view(-1), label.float())
            crf_loss = multiscale_gated_crf_loss(p_student, center_img, kernel=args.crf_kernel,
                                                 dilations=args.crf_dilations,
                                                 sigma_int=args.crf_sigma_int,
                                                 sigma_xy=args.crf_sigma_xy) \
                if args.crf_weight > 0 else logits.new_zeros(())
            spa_loss = spa(ad_center) if args.spa_weight > 0 else logits.new_zeros(())
            rcsa_loss = rcsa(ad_center, soft) if args.rcsa_weight > 0 else logits.new_zeros(())
            # FNFD: pull NEGATIVE-slice features toward the global healthy
            # consensus (cross-client FP suppression), weighted up for FLAIR-poor clients
            fnfd_loss = healthy_distill_loss(feat, label, mu_healthy) \
                if args.fnfd_weight > 0 else logits.new_zeros(())

            loss = (seg_loss + args.proto_weight * proto_loss + args.cons_weight * cons_loss
                    + args.cls_weight * cls_loss + args.crf_weight * crf_loss
                    + align_scale * (args.spa_weight * spa_loss + args.rcsa_weight * rcsa_loss)
                    + fnfd_w * args.fnfd_weight * fnfd_loss)
            # NaN guard: never apply a non-finite step, since a single non-finite
            # client poisons every other client through FedAvg. Skip the batch and
            # keep student/teacher/adapter clean.
            if not torch.isfinite(loss):
                nan_batches += 1
                opt.zero_grad(set_to_none=True)
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), args.grad_clip)  # tame gradient spikes
            opt.step()
            ema_update(teacher, student, args.ema_decay)

            with torch.no_grad():
                lp = local_prototypes(feat.detach(), (soft > 0.5).float(), weight)
                if lp is not None:
                    fv, fc, bv, bc = lp
                    fg_sum += fv * fc; bg_sum += bv * bc; fg_cnt += fc; bg_cnt += bc
                if args.spa_weight > 0:
                    spa_stats.append(spa.compute_local_statistics(ad_center.detach()))
                if args.rcsa_weight > 0:
                    rcsa_stats.append(rcsa.compute_local_statistics(ad_center.detach(), soft))
                hp = healthy_feat_pool(feat.detach(), label)      # healthy-feature accumulator
                if hp is not None:
                    hv, hc = hp; h_sum += hv * hc; h_cnt += hc
            tot['seg'] += float(seg_loss); tot['proto'] += float(proto_loss)
            tot['cls'] += float(cls_loss); tot['crf'] += float(crf_loss)
            tot['spa'] += float(spa_loss); tot['rcsa'] += float(rcsa_loss)
            tot['fnfd'] += float(fnfd_loss)
            weight_accum += float(weight.mean()); n_batches += 1

    n_batches = max(n_batches, 1)
    local_proto = None
    if float(fg_cnt) > 0 and float(bg_cnt) > 0:
        local_proto = {'fg': (fg_sum / fg_cnt).detach(), 'fg_cnt': float(fg_cnt),
                       'bg': (bg_sum / bg_cnt).detach(), 'bg_cnt': float(bg_cnt)}

    def _avg_stats(lst):
        if not lst:
            return None
        keys = set().union(*[s.keys() for s in lst])
        out = {}
        for k in keys:
            vals = [s[k] for s in lst if k in s]
            if vals:
                out[k] = torch.stack(vals).mean(0)
        return out

    finite = all(torch.isfinite(p).all().item() for p in unet.parameters())
    healthy_vec = (h_sum / h_cnt).detach() if h_cnt > 0 else None
    return {'unet': unet, 'reliability': weight_accum / n_batches, 'local_proto': local_proto,
            'spa_stat': _avg_stats(spa_stats), 'rcsa_stat': _avg_stats(rcsa_stats),
            'finite': finite, 'nan_batches': nan_batches,
            'healthy_vec': healthy_vec, 'healthy_cnt': h_cnt,
            **{k: v / n_batches for k, v in tot.items()}}


def federated_train(args):
    device = args.device
    print(f"FedMAP-R v12 (trimap + FEDERATED healthy-tissue feature distillation) | "
          f"clients={args.clients} | device={device}")
    print(f"  clean_targets={args.clean_targets}  crf_dilations={args.crf_dilations} kernel={args.crf_kernel}  "
          f"fnfd_weight={args.fnfd_weight} fnfd_poverty={args.fnfd_poverty}  "
          f"grad_clip={args.grad_clip}  init_best_from={args.init_best_from or 'none (current-run best)'}")
    # FLAIR-absent clients get a larger healthy-distillation weight (they have the
    # weakest healthy/tumor separation and gain most from the FLAIR consensus).
    fnfd_scale = {c: (1.0 + args.fnfd_poverty) if 'flair' not in CLIENT_CONFIG['clients'][str(c)]
                  else 1.0 for c in args.clients}
    print(f"  FNFD per-client weight (FLAIR-absent boosted): {fnfd_scale}")
    t0 = time.time()
    temp = ""
    for c in args.clients:
        temp = f"{c}_" + temp
    save_dir = os.path.join(args.save_dir, temp, "map_refine_v12") + "/"
    os.makedirs(save_dir, exist_ok=True)
    rounds_dir = os.path.join(save_dir, "rounds")
    if args.save_rounds:
        os.makedirs(rounds_dir, exist_ok=True)
    round_val = {c: [] for c in args.clients}   # (round, val_dice) log for the soup
    print(f"Saving models to {save_dir}")

    manifest = pd.read_csv(args.manifest_path)
    split_df = pd.read_csv(args.csv_path)
    context = args.K // 2

    global_unet = build_v4_unet(args.base_filters, args.K, device)
    fdim = global_unet.feat_dim
    adapters, teachers, banks, spas, rcsas = {}, {}, {}, {}, {}
    align_scale = {}
    train_loaders, val_loaders, num_train = {}, {}, []
    lk = {'num_workers': args.num_workers, 'pin_memory': True}
    if args.num_workers > 0:
        lk['persistent_workers'] = True; lk['prefetch_factor'] = 2

    mods = {c: num_in_channels(c) for c in args.clients}
    max_mods = max(mods.values())

    warm = False
    for client in args.clients:
        in_ch = mods[client]
        adapters[client] = ModalityAdapter(in_channels=in_ch).to(device)
        if args.init_from_v4:
            a_path = os.path.join(args.init_from_v4, f"_personalized_modality_client_{client}_best.pth")
            u_path = os.path.join(args.init_from_v4, f"_personalized_seg_unet_client_{client}_best.pth")
            if os.path.exists(a_path):
                adapters[client].load_state_dict(torch.load(a_path, map_location=device)['model_state_dict'])
            if os.path.exists(u_path):
                v4sd = torch.load(u_path, map_location=device)['model_state_dict']
                # v4 already has 3K input conv -> load directly (strict=False safe)
                tmp = build_v4_unet(args.base_filters, args.K, device)
                miss, unexp = tmp.load_state_dict(v4sd, strict=False)
                warm = True
                if client == args.clients[0]:
                    global_unet.load_state_dict(tmp.state_dict())
        teachers[client] = SegPipeline25D(copy.deepcopy(adapters[client]),
                                          copy.deepcopy(global_unet), args.K).to(device)
        for p in teachers[client].parameters():
            p.requires_grad_(False)
        banks[client] = PrototypeBank(fdim, tau=args.proto_tau).to(device)
        spas[client] = SpectralAlign(num_channels=3, num_freq_bands=args.spa_bands).to(device)
        rcsas[client] = RegionConditionedSpectralAlign(num_channels=3, num_freq_bands=args.spa_bands).to(device)
        align_scale[client] = 1.0 + args.poverty_scale * (max_mods - in_ch)
        print(f"  client {client}: in_ch={in_ch} align_scale={align_scale[client]:.1f}")

        m_train = manifest[(manifest['client_id'] == client) & (manifest['split'] == 'train')]
        tr = SegTrainDataset25DTrimap(m_train, args.img_size, CLIENT_CONFIG, client_id=client,
                                      context=context, r_in=args.r_in, r_out=args.r_out,
                                      speckle_min=args.speckle_min, clean=bool(args.clean_targets))
        num_train.append(len(tr))
        train_loaders[client] = DataLoader(tr, batch_size=args.batch_size, shuffle=True, **lk)
        v_df = split_df[(split_df['client_id'] == client) & (split_df['split'] == 'val')]
        va = SegValDataset25D(v_df, args.img_size, CLIENT_CONFIG, client_id=client, context=context)
        val_loaders[client] = DataLoader(va, batch_size=args.batch_size, shuffle=False, **lk)
    print(f"Warm-start from v4: {'YES' if warm else 'NO'}")

    size_w = np.array(num_train, dtype=np.float64)
    best_dice = {c: 0.0 for c in args.clients}
    # ---- best-so-far RATCHET: seed the deployed best + the val threshold from a
    # previous run, so this run can only IMPROVE the per-client deployed model, never
    # regress it, and always has a valid checkpoint even if a fresh round fails to
    # beat the prior best. ----
    if args.init_best_from:
        import shutil
        for client in args.clients:
            for kind in ('_personalized_seg_unet_client', '_personalized_modality_client'):
                src = os.path.join(args.init_best_from, f"{kind}_{client}_best.pth")
                if os.path.exists(src):
                    shutil.copy2(src, os.path.join(save_dir, f"{kind}_{client}_best.pth"))
            sp = os.path.join(args.init_best_from, f"_personalized_seg_unet_client_{client}_best.pth")
            if os.path.exists(sp):
                best_dice[client] = float(torch.load(sp, map_location='cpu').get('val_dice', 0.0))
        print(f"Ratchet: seeded deployed-best from {args.init_best_from}; "
              f"per-client val floor = { {c: round(best_dice[c],4) for c in args.clients} }")
    best_avg = 0.0; patience = 0
    mu_healthy = None   # global healthy-tissue feature prototype (fills from round 1)

    # ---- Round 0: seed spectral prototypes so SPA/RCSA are active from round 1 ----
    if args.spa_weight > 0 or args.rcsa_weight > 0:
        init_w = size_w / size_w.sum()
        ok_spa, ok_rcsa = init_spectral_prototypes(args, adapters, spas, rcsas,
                                                   train_loaders, init_w, device)
        print(f"Round 0: seeded spectral prototypes (SPA={ok_spa}, RCSA={ok_rcsa}) "
              f"-> alignment active from round 1")

    for rnd in range(args.num_rounds):
        if rnd % 5 == 0:
            torch.cuda.empty_cache()
        alpha = alpha_schedule(rnd, args.warmup_rounds, args.ramp_rounds, args.alpha_max)
        use_model = rnd >= args.warmup_rounds
        print(f"\n{'='*60}\nRound {rnd+1}/{args.num_rounds}  alpha={alpha:.2f}\n{'='*60}")
        client_unets, reliabilities, local_protos, dices = [], [], [], []
        spa_all, rcsa_all = [], []
        finite_flags = []; healthy_vecs, healthy_cnts = [], []
        for client in args.clients:
            res = train_one_client(args, client, adapters[client], teachers[client], banks[client],
                                   spas[client], rcsas[client], align_scale[client], global_unet,
                                   train_loaders[client], alpha, use_model, mu_healthy, fnfd_scale[client], device)
            client_unets.append(res['unet']); reliabilities.append(res['reliability'])
            local_protos.append(res['local_proto']); spa_all.append(res['spa_stat']); rcsa_all.append(res['rcsa_stat'])
            finite_flags.append(res['finite'])
            healthy_vecs.append(res['healthy_vec']); healthy_cnts.append(res['healthy_cnt'])
            vdice = validate(adapters[client], res['unet'], val_loaders[client], device,
                             args.gate_tau, args.K, tta=args.tta)
            dices.append(vdice)
            tag = "" if res['finite'] else "  [NON-FINITE unet -> excluded from FedAvg]"
            print(f"  client {client}: seg={res['seg']:.3f} crf={res['crf']:.3f} cls={res['cls']:.3f} "
                  f"fnfd={res['fnfd']:.4f} nan_batches={res['nan_batches']} "
                  f"val_dice(GT)={vdice:.4f}{tag}")
            # only checkpoint FINITE weights with a FINITE val score
            ok = res['finite'] and np.isfinite(vdice)
            if args.save_rounds and ok and vdice > args.round_min_val:
                torch.save({'round': rnd, 'model_state_dict': res['unet'].state_dict(), 'val_dice': vdice, 'K': args.K},
                           os.path.join(rounds_dir, f"seg_client_{client}_r{rnd}.pth"))
                torch.save({'round': rnd, 'model_state_dict': adapters[client].state_dict(), 'val_dice': vdice},
                           os.path.join(rounds_dir, f"adapter_client_{client}_r{rnd}.pth"))
                round_val[client].append((int(rnd), float(vdice)))
            if ok and vdice > best_dice[client]:
                best_dice[client] = vdice
                torch.save({'round': rnd, 'model_state_dict': res['unet'].state_dict(), 'val_dice': vdice, 'K': args.K},
                           os.path.join(save_dir, f"_personalized_seg_unet_client_{client}_best.pth"))
                torch.save({'round': rnd, 'model_state_dict': adapters[client].state_dict(), 'val_dice': vdice},
                           os.path.join(save_dir, f"_personalized_modality_client_{client}_best.pth"))
                print(f"    -> new best client {client} ({vdice:.4f})")

        avg_dice = float(np.mean(dices))
        print(f"\nAverage val Dice(GT): {avg_dice:.4f}")
        # Aggregate ONLY finite clients: a single non-finite client would poison the
        # whole global backbone via FedAvg. Non-finite clients are DROPPED from every
        # aggregation (a zero weight would still give 0*NaN=NaN in the weighted sum).
        # If all are non-finite, the prior global model is kept unchanged this round.
        rel = np.clip(np.array(reliabilities, dtype=np.float64), 1e-3, None)
        fin = np.array(finite_flags, dtype=bool)
        keep = [i for i in range(len(fin)) if fin[i]]
        if keep:
            kept_w = (size_w[keep] * rel[keep]); kept_w = kept_w / kept_w.sum()
            global_unet = aggregate_unets([client_unets[i] for i in keep], kept_w, device)
            if not fin.all():
                print(f"  [FedAvg used {len(keep)}/{len(fin)} finite clients this round]")
            agg_proto = aggregate_prototypes([local_protos[i] for i in keep], device)
            if agg_proto is not None:
                gfg, gbg = agg_proto
                for client in args.clients:
                    banks[client].set_global(gfg, gbg)
            spa_g = aggregate_freq_stats([spa_all[i] for i in keep], kept_w)
            if spa_g is not None:
                for client in args.clients:
                    spas[client].update_global_statistics(spa_g)
            rcsa_g = aggregate_freq_stats([rcsa_all[i] for i in keep], kept_w)
            if rcsa_g is not None:
                for client in args.clients:
                    rcsas[client].update_global_statistics(rcsa_g)
            # Aggregate the global healthy-tissue prototype (count-weighted mean of
            # finite clients' negative-slice features -> broadcast for the next round)
            hv = [(healthy_vecs[i], healthy_cnts[i]) for i in keep
                  if healthy_vecs[i] is not None and healthy_cnts[i] > 0]
            if hv:
                tot_c = sum(c for _, c in hv)
                mu_healthy = sum(v * c for v, c in hv) / tot_c
                print(f"  [FNFD: global healthy prototype updated from {len(hv)} clients]")
        else:
            print("  [ALL clients non-finite this round -> global model + banks unchanged]")

        if avg_dice > best_avg + 1e-4:
            best_avg = avg_dice; patience = 0; print("  New best average. Reset.")
        else:
            patience += 1; print(f"  No improvement. Patience {patience}/{args.patience}")
            if patience >= args.patience:
                print("\nEarly stopping."); break

    print("\nTraining completed!")
    for c in args.clients:
        print(f"  client {c}: best val_dice(GT)={best_dice[c]:.4f}")
    print(f"  mean best val_dice(GT)={np.mean([best_dice[c] for c in args.clients]):.4f}")
    if args.save_rounds:
        import json
        with open(os.path.join(save_dir, "round_val.json"), "w") as f:
            json.dump({str(c): round_val[c] for c in args.clients}, f)
        print(f"  saved per-round val log for soup -> {os.path.join(save_dir, 'round_val.json')}")
    print(f"Total time: {(time.time()-t0)/60:.2f} min")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    # Trimap target-denoising
    ap.add_argument('--clean_targets', type=int, default=1, help='1=trimap denoise, 0=raw CAM')
    ap.add_argument('--r_in', type=int, default=2, help='erosion radius -> confident FG core')
    ap.add_argument('--r_out', type=int, default=4, help='dilation radius -> confident BG start')
    ap.add_argument('--speckle_min', type=int, default=10, help='min CC size kept (px)')
    ap.add_argument('--save_rounds', type=int, default=1, help='save per-round ckpts for the soup')
    ap.add_argument('--round_min_val', type=float, default=0.60, help='only checkpoint rounds above this val')
    ap.add_argument('--grad_clip', type=float, default=10.0, help='max grad norm (NaN guard)')
    ap.add_argument('--init_best_from', type=str, default='',
                    help='seed deployed-best ckpts + per-client val floor from this dir (ratchet)')
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
    ap.add_argument('--K', type=int, default=3)
    ap.add_argument('--init_from_v4', type=str, default='')
    ap.add_argument('--tta', action='store_true')

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
    ap.add_argument('--crf_kernel', type=int, default=3, help='per-scale window (multi-scale CRF)')
    ap.add_argument('--crf_dilations', type=str, default='1,2,4',
                    help='multi-scale gated-CRF dilations; "1" == single-scale')
    ap.add_argument('--fnfd_weight', type=float, default=0.5,
                    help='federated healthy-tissue feature distillation (0 disables)')
    ap.add_argument('--fnfd_poverty', type=float, default=1.0,
                    help='extra FNFD weight for FLAIR-absent clients')
    ap.add_argument('--crf_sigma_int', type=float, default=0.15)
    ap.add_argument('--crf_sigma_xy', type=float, default=6.0)
    ap.add_argument('--gate_tau', type=float, default=0.5)
    # Spectral alignment
    ap.add_argument('--spa_weight', type=float, default=0.1, help='whole-map spectral alignment')
    ap.add_argument('--rcsa_weight', type=float, default=0.1, help='region-conditioned spectral alignment (novel)')
    ap.add_argument('--spa_bands', type=int, default=8)
    ap.add_argument('--poverty_scale', type=float, default=1.0,
                    help='extra alignment weight per missing modality')
    ap.add_argument('--log_dir', type=str, default='nohups')
    ap.add_argument('--log_file', type=str, default=None)
    args = ap.parse_args()

    setup_tee_logging(args.log_dir, args.log_file)
    args.clients = [int(c) for c in args.clients.split(',')]
    args.crf_dilations = tuple(int(d) for d in str(args.crf_dilations).split(',') if d != '')
    if str(args.device).startswith('cuda') and not torch.cuda.is_available():
        args.device = 'cpu'
    federated_train(args)
