"""
This version includes:
1. Statistical prototype-based alignment losses
2. Early stopping with patience

Key Features:
- Trains aggregation network (Res_Scoring) + local modality models (SimpleUNet)
- Uses client-specific pretrained models (frozen) for supervision
- Adds statistical prototype alignment to encourage similar features across clients
- Server aggregates both aggregation network and statistical prototypes

"""

import torch
import torch.nn as nn
import argparse
import ast
import gc
from datetime import datetime
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
import os
import sys
from tqdm import tqdm
import copy
from time import sleep
import time
import pandas as pd
import numpy as np
import random
import json
from dataset_personalized_2 import ImageDataset
from models_unet_scoring import *
from loss import *
from sklearn.metrics import roc_auc_score, confusion_matrix
from federated_modality_alignment_losses import (
    FeatureStatisticsAlignmentLoss,
    PrototypeAlignmentLoss,
    CombinedModalityAlignmentLoss,
    HigherOrderStatisticsAlignmentLoss,
    HistogramPrototypeAlignmentLoss,
    SpectralPrototypeAlignmentLoss
)

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
torch.backends.cudnn.deterministic = True


def setup_tee_logging(log_dir=None, log_file=None):
    script_name = os.path.splitext(os.path.basename(__file__))[0] if "__file__" in globals() else "train"
    if log_file is None:
        if log_dir is None:
            log_dir = "nohups"
        os.makedirs(log_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(log_dir, f"{script_name}_{timestamp}.log")
    else:
        log_dir = os.path.dirname(log_file)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)

    log_handle = open(log_file, "a", buffering=1)
    stdout = sys.stdout
    stderr = sys.stderr

    class TeeStream:
        def __init__(self, *streams):
            self.streams = streams

        def write(self, data):
            for stream in self.streams:
                stream.write(data)
            self.flush()

        def flush(self):
            for stream in self.streams:
                stream.flush()

    sys.stdout = TeeStream(stdout, log_handle)
    sys.stderr = TeeStream(stderr, log_handle)
    print(f"Logging to {log_file}")
    return log_file


class AggNetWithStatisticalPrototypes(object):
    """Aggregation network training with statistical prototype alignment"""

    def __init__(self, args, model, modality_model, alignment_loss, train_loader, val_loader, 
                 client_id, pretrained_models, alignment_weight=0.1, modality_optimizer=None,
                 map_alignment_loss=None, map_alignment_weight=0.1, fed_method='fedavg', mu=0.01, temperature=0.5, global_agg_net=None, prev_unet=None):
        """
        Args:
            alignment_loss: Statistical prototype alignment loss module for modality features
            alignment_weight: Weight for modality alignment loss term
            map_alignment_loss: Statistical prototype alignment loss module for map_att features
            map_alignment_weight: Weight for map_att alignment loss term
        """
        self.epochs = args.epochs
        self.batch_size = args.batch_size
        self.num_classes = args.num_classes
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.client_id = client_id
        self.task = args.task
        self.lr = args.learning_rate
        self.model = model
        self.modality_model = modality_model
        self.alignment_loss = alignment_loss
        self.alignment_weight = alignment_weight
        self.map_alignment_loss = map_alignment_loss
        self.map_alignment_weight = map_alignment_weight
        self.device = args.device
        self.loss_weight = args.loss_weight if args.loss_weight is not None else [1.0, 1.0, 1.0, 1.0, 5.0]
        self.fed_method = fed_method
        self.mu = mu
        self.temperature = temperature
        self.global_agg_net = global_agg_net
        self.prev_unet = prev_unet
        self.use_amp = getattr(args, 'amp', False)
        self.amp_dtype = torch.float16 if getattr(args, 'amp_dtype', 'float16') == 'float16' else torch.bfloat16
        self.amp_enabled = self.use_amp and str(self.device).startswith("cuda")
        self.scaler = GradScaler(enabled=self.amp_enabled)

        # Soft gating alpha: when > 0, keeps alpha fraction of signal even when binary gate is off
        self.soft_gating_alpha = getattr(args, 'soft_gating_alpha', 0.0)

        # Use pre-loaded pretrained models
        if self.task == "binary":
            self.binary_model = pretrained_models['binary_model']
            self.binary_modality_model = pretrained_models['binary_modality_model']
        else:
            self.binary_model = pretrained_models['binary_model']
            self.binary_modality_model = pretrained_models['binary_modality_model']
            self.multi_model = pretrained_models['multi_model']
            self.multi_modality_model = pretrained_models['multi_modality_model']
            self.bin_score_model = pretrained_models['bin_score_model']
            self.bin_score_modality_model = pretrained_models['bin_score_modality_model']

        self.score_optimizer = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=1e-5)
        if modality_optimizer is None:
            self.modality_optimizer = optim.Adam(self.modality_model.parameters(), lr=self.lr, weight_decay=1e-5)
        else:
            self.modality_optimizer = modality_optimizer

        self.loss = nn.BCEWithLogitsLoss().to(self.device)
        self.sminloss_intra = SimMinLoss().to(self.device)
        self.sminloss_inter = SimMinLoss(intra=False).to(self.device)
        self.smaxloss = SimMaxLoss_intraclass().to(self.device)
        use_per_class = getattr(args, 'per_class_agreement', False)
        self.aggrementloss = AggrementLoss(per_class=use_per_class).to(self.device)

    def train(self, epoch, collect_stats=False):
        """Train with statistical prototype alignment"""
        self.model.train()
        self.modality_model.train()

        train_bar = self.train_loader
        total_loss, total_num = 0.0, 0
        total_agg_loss, total_align_loss, total_map_align_loss = 0.0, 0.0, 0.0
        clloss_epoch = {}
        stats_to_aggregate = []
        map_stats_to_aggregate = []

        for epoch in range(1, self.epochs + 1):
            for idx, (case_batch, label_batch) in enumerate(train_bar):
                case_batch = case_batch.to(self.device, non_blocking=True)
                label_batch = label_batch.to(self.device, non_blocking=True)
                
                # Zero gradients
                self.score_optimizer.zero_grad(set_to_none=True)
                self.modality_optimizer.zero_grad(set_to_none=True)
                
                with autocast(enabled=self.amp_enabled, dtype=self.amp_dtype):
                    # Get predictions from pretrained models
                    pred_binary, pred_multi, map_collect_binary, map_collect_multi = self.step(case_batch, label_batch)
                
                    # Forward through local modality model
                    modality_features = self.modality_model(case_batch)
                    
                    # Aggregation network loss and get map_att features
                    if self.map_alignment_loss is not None:
                        agg_loss, clloss_collect, map_att_for_alignment, map_collect_moon = self.score_step(
                            case_batch, modality_features, label_batch, map_collect_binary, map_collect_multi, return_map_att=True)
                    else:
                        agg_loss, clloss_collect = self.score_step(
                            case_batch, modality_features, label_batch, map_collect_binary, map_collect_multi)
                    
                    # Statistical prototype alignment loss on modality features
                    if isinstance(self.alignment_loss, CombinedModalityAlignmentLoss):
                        align_loss, _ = self.alignment_loss(modality_features)
                    else:
                        align_loss = self.alignment_loss(modality_features)
                    
                    # Map attention alignment loss (if enabled)
                    map_align_loss = 0
                    if self.map_alignment_loss is not None:
                        if isinstance(self.map_alignment_loss, CombinedModalityAlignmentLoss):
                            map_align_loss, _ = self.map_alignment_loss(map_att_for_alignment)
                        else:
                            map_align_loss = self.map_alignment_loss(map_att_for_alignment)
                            
                    
                    # Combined loss
                    total_loss_batch = agg_loss + self.alignment_weight * align_loss
                    if self.map_alignment_loss is not None:
                        total_loss_batch = total_loss_batch + self.map_alignment_weight * map_align_loss
                    prox_loss = 0
                    if self.fed_method == 'fedprox' and self.global_agg_net is not None:
                        for w, w_g in zip(self.model.parameters(), self.global_agg_net.parameters()):
                            prox_loss += (w - w_g).norm(2)
                        prox_loss = (self.mu / 2) * prox_loss
                        total_loss_batch = total_loss_batch + prox_loss
                    if self.fed_method == 'moon' and self.global_agg_net is not None:
                        with torch.no_grad():
                            self.global_agg_net.eval()
                            global_modality_features = modality_features.detach()
                            map_global_agg = self.global_agg_net(global_modality_features, map_collect_moon, return_map_att=False, moon=True)
                            
                            if self.prev_unet is not None:
                                self.prev_unet.eval()
                                prev_l1 = self.prev_unet(global_modality_features, map_collect_moon, return_map_att=False, moon=True)
                    
                        # Use l1 features as representations (64 channels, spatially pooled)
                        local_rep = F.adaptive_avg_pool2d(map_att_for_alignment, (1, 1)).flatten(1)
                        global_rep = F.adaptive_avg_pool2d(map_global_agg, (1, 1)).flatten(1)

                        # Normalize representations
                        local_rep = F.normalize(local_rep, dim=1)
                        global_rep = F.normalize(global_rep, dim=1)

                        # Positive similarity (with global model)
                        pos_sim = torch.exp(torch.sum(local_rep * global_rep, dim=1) / self.temperature)

                        # Negative similarity (with previous local model if available)
                        if self.prev_unet is not None:
                            prev_rep = F.adaptive_avg_pool2d(prev_l1, (1, 1)).flatten(1)
                            prev_rep = F.normalize(prev_rep, dim=1)
                            neg_sim = torch.exp(torch.sum(local_rep * prev_rep, dim=1) / self.temperature)
                            moon_loss = -torch.log(pos_sim / (pos_sim + neg_sim)).mean()
                        else:
                            # If no previous model, just maximize similarity with global
                            moon_loss = -torch.log(pos_sim / (pos_sim + 1e-8)).mean()
                        
                        total_loss_batch = total_loss_batch + moon_loss
                
                self.scaler.scale(total_loss_batch).backward()
                
                # Update both models
                self.scaler.step(self.score_optimizer)
                self.scaler.step(self.modality_optimizer)
                self.scaler.update()

                total_num += case_batch.size(0)
                total_loss += total_loss_batch.item() * case_batch.size(0)
                total_agg_loss += agg_loss.item() * case_batch.size(0)
                total_align_loss += align_loss.item() * case_batch.size(0)
                if self.map_alignment_loss is not None:
                    total_map_align_loss += map_align_loss.item() * case_batch.size(0)
                
                for k, v in {**clloss_collect}.items():
                    clloss_epoch[k] = (clloss_epoch.get(k, 0)) + v.item() * case_batch.size(0)
                
                # Collect statistics for aggregation
                if collect_stats:
                    with torch.no_grad():
                        if hasattr(self.alignment_loss, 'compute_local_statistics'):
                            local_stats = self.alignment_loss.compute_local_statistics(modality_features)
                            if local_stats is not None:
                                stats_to_aggregate.append(local_stats)
                        
                        if self.map_alignment_loss is not None and hasattr(self.map_alignment_loss, 'compute_local_statistics'):
                            map_local_stats = self.map_alignment_loss.compute_local_statistics(map_att_for_alignment)
                            if map_local_stats is not None:
                                map_stats_to_aggregate.append(map_local_stats)
            
            for k in clloss_epoch.keys():
                clloss_epoch[k] = clloss_epoch[k] / total_num

        avg_stats = None
        if stats_to_aggregate:
            # Average statistics over all batches
            avg_stats = {}
            stat_keys = stats_to_aggregate[0].keys()
            for key in stat_keys:
                values = [s[key] for s in stats_to_aggregate if key in s]
                if values:
                    avg_stats[key] = torch.stack(values).mean(dim=0)
        
        avg_map_stats = None
        if map_stats_to_aggregate:
            # Average map statistics over all batches
            avg_map_stats = {}
            stat_keys = map_stats_to_aggregate[0].keys()
            for key in stat_keys:
                values = [s[key] for s in map_stats_to_aggregate if key in s]
                if values:
                    avg_map_stats[key] = torch.stack(values).mean(dim=0)

        return {
            'total_loss': total_loss / total_num,
            'agg_loss': total_agg_loss / total_num,
            'alignment_loss': total_align_loss / total_num,
            'map_alignment_loss': total_map_align_loss / total_num if self.map_alignment_loss is not None else 0,
            'statistics': avg_stats,
            'map_statistics': avg_map_stats
        }

    def step(self, case_batch, label_batch):
        """Pass through client-specific pretrained models (frozen)"""
        with torch.no_grad():
            binary_features = self.binary_modality_model(case_batch)
            logits_collect_binary, map_collect_binary = self.binary_model(binary_features)
        pred_binary = torch.sigmoid(logits_collect_binary[-1])

        if self.task == "binary":
            return pred_binary.cpu(), None, map_collect_binary, None
        else:
            with torch.no_grad():
                multi_features = self.multi_modality_model(case_batch)
                logits_collect_multi, map_collect_multi = self.multi_model(multi_features)
            pred_multi = torch.sigmoid(logits_collect_multi[-1])
            return pred_binary.cpu(), pred_multi.cpu(), map_collect_binary, map_collect_multi

    def normalize_map(self, tensor):
        a1, a2, a3, a4 = tensor.size()
        tensor = tensor.view(a1, a2, -1)
        tensor_min = tensor.min(2, keepdim=True)[0]
        tensor_max = tensor.max(2, keepdim=True)[0]
        tensor = (tensor - tensor_min) / (tensor_max - tensor_min + 1e-5)
        tensor = tensor.view(a1, a2, a3, a4)
        return tensor
    
    def postprocess_cam(self, cam, binary_cam, thresholds=None):
        """Post-process CAM with binary mask (same as test time)"""
        cam = np.maximum(cam, 0)
        cam = cam / (cam.max() + 1e-8)
        binary_cam = np.maximum(binary_cam, 0)
        binary_cam = binary_cam / (binary_cam.max() + 1e-8)

        cam = np.where(binary_cam > thresholds, cam * binary_cam, 0)
        cam = np.where(cam > 0.4, 1, 0)

        return cam
    
    def CAM_algo(self, ame_map):
        """Process CAM maps (same as test time)"""
        for i in range(ame_map.shape[1]):
            if (ame_map[0][i].max() - ame_map[0][i].min()) > 0:
                ame_map[0][i] = (ame_map[0][i] - ame_map[0][i].min()) / (ame_map[0][i].max() - ame_map[0][i].min() + 1e-5)

        ame_map = ame_map.squeeze(0).numpy()
        ame_map = (1 - ame_map)
        return ame_map

    def score_step(self, case_batch, modality_features, label_batch, map_collect_binary, map_collect_multi, return_map_att=False, return_final_maps=False):
        """Train aggregation network and optionally return map_att features for alignment or final processed CAM maps"""
        loss = 0
        loss_collect = {}
        map_att_for_alignment = None
        final_cam_maps = None
        map_collect_1 = map_collect_binary.copy()
        if self.task == "binary":
            if return_map_att:
                _, foreground, background, binary_ame_map, map_att_for_alignment = self.model(
                    modality_features, map_collect_binary, return_map_att=True)
            else:
                _, foreground, background, binary_ame_map = self.model(modality_features, map_collect_binary)
            loss_collect["SimMax_Foreground_intra"] = (self.loss_weight[0] * self.smaxloss(foreground))
            loss_collect["SimMax_background_intra"] = (self.loss_weight[1] * self.smaxloss(background))
            loss_collect["SimMin_intra_foreground_background"] = (self.loss_weight[2] * self.sminloss_intra(foreground, background))
        else:
            with torch.no_grad():
                bin_score_features = self.bin_score_modality_model(case_batch)
                _, binary_foreground, binary_background, binary_ame_map = self.bin_score_model(bin_score_features, map_collect_binary)

            binary_ame_map = self.normalize_map(binary_ame_map)
            map_collect_multi = torch.stack(map_collect_multi, dim=0)
            # Soft binary gating with configurable alpha leakage
            if self.soft_gating_alpha > 0:
                map_collect = (self.soft_gating_alpha + (1 - self.soft_gating_alpha) * binary_ame_map) * map_collect_multi
            else:
                map_collect = binary_ame_map * map_collect_multi
            map_collect = map_collect.unbind(0)
            map_collect = [map_collect[i] for i in range(len(map_collect))]
            map_collect_1 = map_collect.copy()
            if return_map_att:
                _, foreground, background, multi_ame_map, map_att_for_alignment = self.model(
                    modality_features, map_collect, return_map_att=True)
            else:
                _, foreground, background, multi_ame_map = self.model(modality_features, map_collect)
        
            loss_collect["SimMax_Foreground_intra"] = (self.loss_weight[0] * self.smaxloss(foreground))
            loss_collect["SimMax_background_intra"] = (self.loss_weight[1] * self.smaxloss(background))
            loss_collect["SimMin_intra_foreground_background"] = (self.loss_weight[2] * self.sminloss_intra(foreground, background))
            loss_collect["SimMMin_inter_foreground"] = (self.loss_weight[3] * self.sminloss_inter(foreground, foreground))
            loss_collect["Aggrement_loss"] = (self.loss_weight[4] * self.aggrementloss(binary_ame_map, multi_ame_map))
            
            # Process final CAM maps if requested (for multiclass only)
            if return_final_maps:
                # Concatenate multi and binary attention maps (same as test time)
                combined_ame_map = torch.cat((multi_ame_map, binary_ame_map), dim=1)
                # Process with CAM_algo
                pseudo_labels = torch.zeros(case_batch.size(0), self.num_classes, 224, 224)
                ame_map = self.CAM_algo(combined_ame_map.detach().cpu())
                for i, class_name in enumerate(['core', 'edema']):
                    cam_map = ame_map[i]
                    processed_cam = self.postprocess_cam(ame_map[i], ame_map[-1], thresholds=0.5)
                    pseudo_labels[:, i, :, :] = torch.tensor(processed_cam)

        for k, l in loss_collect.items():
            loss += l

        if return_map_att and return_final_maps:
            return loss, loss_collect, map_att_for_alignment, final_cam_maps
        elif return_map_att:
            return loss, loss_collect, map_att_for_alignment, map_collect_1
        elif return_final_maps:
            return loss, loss_collect, final_cam_maps
        return loss, loss_collect
    
    def val(self):
        """Validation"""
        self.model.eval()
        self.modality_model.eval()
            
        val_bar = self.val_loader
        total_loss, total_num = 0.0, 0
        clloss_epoch = {}

        with torch.no_grad():
            for idx, (case_batch, label_batch) in enumerate(val_bar):
                case_batch = case_batch.to(self.device, non_blocking=True)
                label_batch = label_batch.to(self.device, non_blocking=True)

                with autocast(enabled=self.amp_enabled, dtype=self.amp_dtype):
                    pred_binary, pred_multi, map_collect_binary, map_collect_multi = self.step(case_batch, label_batch)
                    modality_features = self.modality_model(case_batch)
                    clloss, clloss_collect = self.score_step(case_batch, modality_features, label_batch, 
                                                              map_collect_binary, map_collect_multi)

                total_num += case_batch.size(0)
                total_loss += clloss.item() * case_batch.size(0)

                for k, v in clloss_collect.items():
                    clloss_epoch[k] = clloss_epoch.get(k, 0.0) + v.item() * case_batch.size(0)

        for k in clloss_epoch:
            clloss_epoch[k] /= total_num

        return total_loss / total_num


def aggregate_models(client_models, weights, device="cpu", global_model_prev=None, 
                    current_round=0, cosine_agg_method=None, cosine_start_round=11):
    """
    Aggregate model parameters using FedAvg or cosine similarity-based methods
    
    Args:
        client_models: List of client models
        weights: Client weights (data-based or uniform)
        device: Device to use
        global_model_prev: Previous round's global model (for cosine similarity)
        current_round: Current training round
        cosine_agg_method: Method for cosine similarity aggregation
            - None: Standard FedAvg
            - 'layer_wise': Layer-wise cosine similarity
            - 'filter_wise_client': Filter-wise cosine per client
            - 'filter_wise_global': Filter-wise cosine global normalization
        cosine_start_round: Round to start using cosine similarity (default: 11)
    """
    global_model = copy.deepcopy(client_models[0])
    global_model.to(device)
    global_model.train()

    # Standard FedAvg if before start round or no cosine method specified
    if current_round < cosine_start_round or cosine_agg_method is None or global_model_prev is None:
        for global_param in global_model.parameters():
            global_param.data = torch.zeros_like(global_param.data)

        for client_model, weight in zip(client_models, weights):
            for global_param, client_param in zip(global_model.parameters(), client_model.parameters()):
                global_param.data += weight * client_param.data

        return global_model
    
    # Cosine similarity-based aggregation
    if cosine_agg_method == 'layer_wise':
        return _aggregate_layer_wise_cosine(client_models, weights, global_model, global_model_prev, device)
    elif cosine_agg_method == 'filter_wise_client':
        return _aggregate_filter_wise_client_cosine(client_models, weights, global_model, global_model_prev, device)
    elif cosine_agg_method == 'filter_wise_global':
        return _aggregate_filter_wise_global_cosine(client_models, weights, global_model, global_model_prev, device)
    else:
        raise ValueError(f"Unknown cosine_agg_method: {cosine_agg_method}")


def _aggregate_layer_wise_cosine(client_models, data_weights, global_model, global_model_prev, device):
    """Layer-wise Cosine Similarity Aggregation"""
    print(f"  Using Layer-wise Cosine Similarity Aggregation")
    
    for global_param in global_model.parameters():
        global_param.data = torch.zeros_like(global_param.data)
    
    param_names = [name for name, _ in global_model.named_parameters()]
    
    for param_idx, (global_param, prev_param) in enumerate(zip(
        global_model.parameters(), 
        global_model_prev.parameters()
    )):
        param_name = param_names[param_idx]
        
        # Skip classification head
        if 'classifier' in param_name.lower() or 'fc' in param_name.lower() or 'head' in param_name.lower():
            for client_model, weight in zip(client_models, data_weights):
                client_param = list(client_model.parameters())[param_idx]
                global_param.data += weight * client_param.data
            continue
        
        # Compute cosine similarity for this layer
        layer_cosine_weights = []
        for client_model in client_models:
            client_param = list(client_model.parameters())[param_idx]
            client_flat = client_param.data.flatten()
            prev_flat = prev_param.data.flatten()
            
            cos_sim = torch.nn.functional.cosine_similarity(
                client_flat.unsqueeze(0), 
                prev_flat.unsqueeze(0), 
                dim=1
            ).item()
            
            layer_cosine_weights.append(max(cos_sim, 0.0))
        
        # Normalize
        layer_cosine_weights = torch.tensor(layer_cosine_weights, device=device)
        if layer_cosine_weights.sum() > 0:
            layer_cosine_weights = layer_cosine_weights / layer_cosine_weights.sum()
        else:
            layer_cosine_weights = torch.tensor(data_weights, device=device)
        
        # Aggregate with layer-specific weights
        for client_idx, client_model in enumerate(client_models):
            client_param = list(client_model.parameters())[param_idx]
            global_param.data += layer_cosine_weights[client_idx] * client_param.data
    
    return global_model


def _aggregate_filter_wise_client_cosine(client_models, data_weights, global_model, global_model_prev, device):
    """Filter-wise Cosine per Client Aggregation"""
    print(f"  Using Filter-wise Client Cosine Similarity Aggregation")
    
    for global_param in global_model.parameters():
        global_param.data = torch.zeros_like(global_param.data)
    
    param_names = [name for name, _ in global_model.named_parameters()]
    
    for client_idx, client_model in enumerate(client_models):
        weighted_model_params = []
        
        for param_idx, (client_param, prev_param) in enumerate(zip(
            client_model.parameters(),
            global_model_prev.parameters()
        )):
            param_name = param_names[param_idx]
            
            if 'classifier' in param_name.lower() or 'fc' in param_name.lower() or 'head' in param_name.lower():
                weighted_model_params.append(client_param.data.clone())
                continue
            
            if len(client_param.shape) >= 2:
                filter_cosine_weights = []
                
                if len(client_param.shape) == 4:  # Conv
                    num_filters = client_param.shape[0]
                    for filter_idx in range(num_filters):
                        client_filter = client_param.data[filter_idx].flatten()
                        prev_filter = prev_param.data[filter_idx].flatten()
                        
                        cos_sim = torch.nn.functional.cosine_similarity(
                            client_filter.unsqueeze(0),
                            prev_filter.unsqueeze(0),
                            dim=1
                        ).item()
                        filter_cosine_weights.append(max(cos_sim, 0.0))
                    
                    filter_cosine_weights = torch.tensor(filter_cosine_weights, device=device)
                    min_val = filter_cosine_weights.min()
                    max_val = filter_cosine_weights.max()
                    if max_val > min_val:
                        filter_cosine_weights = (filter_cosine_weights - min_val) / (max_val - min_val)
                    else:
                        filter_cosine_weights = torch.ones_like(filter_cosine_weights)
                    
                    weighted_param = client_param.data.clone()
                    for filter_idx in range(num_filters):
                        weighted_param[filter_idx] *= filter_cosine_weights[filter_idx]
                    
                    weighted_model_params.append(weighted_param)
                
                elif len(client_param.shape) == 2:  # Linear
                    num_filters = client_param.shape[0]
                    for filter_idx in range(num_filters):
                        client_filter = client_param.data[filter_idx].flatten()
                        prev_filter = prev_param.data[filter_idx].flatten()
                        
                        cos_sim = torch.nn.functional.cosine_similarity(
                            client_filter.unsqueeze(0),
                            prev_filter.unsqueeze(0),
                            dim=1
                        ).item()
                        filter_cosine_weights.append(max(cos_sim, 0.0))
                    
                    filter_cosine_weights = torch.tensor(filter_cosine_weights, device=device)
                    min_val = filter_cosine_weights.min()
                    max_val = filter_cosine_weights.max()
                    if max_val > min_val:
                        filter_cosine_weights = (filter_cosine_weights - min_val) / (max_val - min_val)
                    else:
                        filter_cosine_weights = torch.ones_like(filter_cosine_weights)
                    
                    weighted_param = client_param.data.clone()
                    for filter_idx in range(num_filters):
                        weighted_param[filter_idx] *= filter_cosine_weights[filter_idx]
                    
                    weighted_model_params.append(weighted_param)
                else:
                    weighted_model_params.append(client_param.data.clone())
            else:
                weighted_model_params.append(client_param.data.clone())
        
        for param_idx, (global_param, weighted_param) in enumerate(zip(
            global_model.parameters(),
            weighted_model_params
        )):
            global_param.data += data_weights[client_idx] * weighted_param
    
    return global_model


def _aggregate_filter_wise_global_cosine(client_models, data_weights, global_model, global_model_prev, device):
    """Filter-wise Cosine Global Normalization Aggregation"""
    print(f"  Using Filter-wise Global Cosine Similarity Aggregation")
    
    for global_param in global_model.parameters():
        global_param.data = torch.zeros_like(global_param.data)
    
    param_names = [name for name, _ in global_model.named_parameters()]
    
    for param_idx, (global_param, prev_param) in enumerate(zip(
        global_model.parameters(),
        global_model_prev.parameters()
    )):
        param_name = param_names[param_idx]
        
        if 'classifier' in param_name.lower() or 'fc' in param_name.lower() or 'head' in param_name.lower():
            for client_idx, client_model in enumerate(client_models):
                client_param = list(client_model.parameters())[param_idx]
                global_param.data += data_weights[client_idx] * client_param.data
            continue
        
        client_params = [list(client_model.parameters())[param_idx] for client_model in client_models]
        
        if len(global_param.shape) >= 2:
            if len(global_param.shape) == 4:  # Conv
                num_filters = global_param.shape[0]
                
                for filter_idx in range(num_filters):
                    filter_cosine_weights = []
                    
                    for client_param in client_params:
                        client_filter = client_param.data[filter_idx].flatten()
                        prev_filter = prev_param.data[filter_idx].flatten()
                        
                        cos_sim = torch.nn.functional.cosine_similarity(
                            client_filter.unsqueeze(0),
                            prev_filter.unsqueeze(0),
                            dim=1
                        ).item()
                        filter_cosine_weights.append(max(cos_sim, 0.0))
                    
                    filter_cosine_weights = torch.tensor(filter_cosine_weights, device=device)
                    min_val = filter_cosine_weights.min()
                    max_val = filter_cosine_weights.max()
                    if max_val > min_val:
                        filter_cosine_weights = (filter_cosine_weights - min_val) / (max_val - min_val)
                    else:
                        filter_cosine_weights = torch.ones_like(filter_cosine_weights)
                    
                    for client_idx, client_param in enumerate(client_params):
                        global_param.data[filter_idx] += (
                            filter_cosine_weights[client_idx] * 
                            data_weights[client_idx] * 
                            client_param.data[filter_idx]
                        )
            
            elif len(global_param.shape) == 2:  # Linear
                num_filters = global_param.shape[0]
                
                for filter_idx in range(num_filters):
                    filter_cosine_weights = []
                    
                    for client_param in client_params:
                        client_filter = client_param.data[filter_idx].flatten()
                        prev_filter = prev_param.data[filter_idx].flatten()
                        
                        cos_sim = torch.nn.functional.cosine_similarity(
                            client_filter.unsqueeze(0),
                            prev_filter.unsqueeze(0),
                            dim=1
                        ).item()
                        filter_cosine_weights.append(max(cos_sim, 0.0))
                    
                    filter_cosine_weights = torch.tensor(filter_cosine_weights, device=device)
                    min_val = filter_cosine_weights.min()
                    max_val = filter_cosine_weights.max()
                    if max_val > min_val:
                        filter_cosine_weights = (filter_cosine_weights - min_val) / (max_val - min_val)
                    else:
                        filter_cosine_weights = torch.ones_like(filter_cosine_weights)
                    
                    for client_idx, client_param in enumerate(client_params):
                        global_param.data[filter_idx] += (
                            filter_cosine_weights[client_idx] * 
                            data_weights[client_idx] * 
                            client_param.data[filter_idx]
                        )
            else:
                for client_idx, client_param in enumerate(client_params):
                    global_param.data += data_weights[client_idx] * client_param.data
        else:
            for client_idx, client_param in enumerate(client_params):
                global_param.data += data_weights[client_idx] * client_param.data
    
    return global_model


def aggregate_statistics(client_stats_list, weights):
    """Aggregate statistics from all clients"""
    if not client_stats_list or all(s is None for s in client_stats_list):
        return None
    
    valid_stats = [(s, w) for s, w in zip(client_stats_list, weights) if s is not None]
    if not valid_stats:
        return None
    
    aggregated = {}
    stat_keys = valid_stats[0][0].keys()
    
    for key in stat_keys:
        values_and_weights = [(s[key], w) for s, w in valid_stats if key in s]
        if values_and_weights:
            weighted_sum = sum(v * w for v, w in values_and_weights)
            total_weight = sum(w for _, w in values_and_weights)
            aggregated[key] = weighted_sum / total_weight
    
    return aggregated


def helper_save_model(round_num, client_id, model, path_prefix, val_loss, local_best=False):
    """Helper to save model checkpoints"""
    if local_best:
        save_path = f"{path_prefix}_client_{client_id}_best.pth"
    else:
        save_path = f"{path_prefix}_round_{round_num}_client_{client_id}.pth"
    
    torch.save({
        'round': round_num,
        'model_state_dict': model.state_dict(),
        'val_loss': val_loss
    }, save_path)


def federated_train_agg_net_statistical_prototypes(args):
    """
    Main federated training function for aggregation network with statistical prototypes
    and advanced aggregation strategies
    """
    print(f"Starting Federated Aggregation Network Training")
    print(f"Clients: {args.clients}")
    print(f"Task: {args.task}")
    print(f"Alignment Type: {args.alignment_type}")
    print(f"Cosine Aggregation Method: {args.cosine_agg_method}")
    print(f"Cosine Start Round: {args.cosine_start_round}")
    
    # Setup
    device = args.device
    start_time = time.time()
    if args.task == 'binary':
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
            "6":["flair", "t1ce"],
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
            "6":["flair", "t1ce"],
        }
        }

    df = pd.read_csv(args.csv_path)
    
    # Create save directory
    temp_path = ""
    for client in args.clients:
        temp_path = str(client) + "_" + temp_path
    model_save_path = args.save_dir + temp_path + "/" + "num_of_bands_" + str(args.num_of_freq_bands) + "/"
    temp_path_1 = temp_path
    print(temp_path_1)
    if args.num_of_freq_bands ==8:
        pass
    else:
        temp_path_1 = f'{temp_path_1}/num_of_bands_{str(args.num_of_freq_bands)}'
    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path, exist_ok=True)

    # Initialize global aggregation network
    use_spatial_norm = getattr(args, 'spatial_normalize', False)
    global_agg_net = Res_Scoring(spatial_normalize=use_spatial_norm).to(device)
    global_agg_net_prev = copy.deepcopy(global_agg_net)  # For cosine similarity
    global_agg_net_prev.eval()
    
    # Load pretrained models for all clients
    print("\nLoading pretrained models...")
    all_pretrained_models = {}
    global_best_val_loss  = float('inf')
    for client in args.clients:
        print(f"Loading models for client {client}...")
        pretrained_models = {}

        # Binary classifier
        bin_path = os.path.join(args.bin_pretrained_dir, temp_path_1, 
                               f'_personalized_unet_client_{client}.pth')
        print(f"Binary model path: {bin_path}")
        binary_model = Res18_Classifier(num_classes=1).to(device)
        binary_model.load_pretrain_weight(bin_path)
        binary_model.eval()
        pretrained_models['binary_model'] = binary_model
        
        # Binary modality
        print(f"Loading binary model for client {client}...")
        bin_mod_path = os.path.join(args.bin_modality_pretrained_dir, temp_path_1,
                                   f'_personalized_modality_client_{client}.pth')
        binary_modality_model = SimpleUNet(in_channels=len(config['clients'][str(client)])).to(device)
        checkpoint = torch.load(bin_mod_path, map_location=device)
        binary_modality_model.load_state_dict(checkpoint['model_state_dict'])
        binary_modality_model.eval()
        pretrained_models['binary_modality_model'] = binary_modality_model
        
        if args.task == "multiclass":
            # Multiclass classifier
            multi_path = os.path.join(args.multi_pretrained_dir, temp_path_1,
                                     f'_personalized_unet_client_{client}.pth')
            multi_model = Res18_Classifier(num_classes=2).to(device)
            multi_model.load_pretrain_weight(multi_path)
            multi_model.eval()
            pretrained_models['multi_model'] = multi_model
            
            # Multiclass modality
            multi_mod_path = os.path.join(args.multi_modality_pretrained_dir, temp_path_1,
                                         f'_personalized_modality_client_{client}.pth')
            multi_modality_model = SimpleUNet(in_channels=len(config['clients'][str(client)])).to(device)
            checkpoint = torch.load(multi_mod_path, map_location=device)
            multi_modality_model.load_state_dict(checkpoint['model_state_dict'])
            multi_modality_model.eval()
            pretrained_models['multi_modality_model'] = multi_modality_model
            
            # Binary score model
            bin_score_path = os.path.join(args.bin_score_pretrained_dir, temp_path_1,"num_of_bands_" + str(args.num_of_freq_bands),
                                         f'_personalized_scoring_client_{client}_best.pth')
            bin_score_model = Res_Scoring(use_unet=True).to(device)
            checkpoint = torch.load(bin_score_path, map_location=device)
            bin_score_model.load_state_dict(checkpoint['model_state_dict'])
            bin_score_model.eval()
            pretrained_models['bin_score_model'] = bin_score_model
            
            # Binary score modality
            bin_score_mod_path = os.path.join(args.bin_score_modality_pretrained_dir, temp_path_1,"num_of_bands_" + str(args.num_of_freq_bands),
                                             f'_personalized_modality_client_{client}_best.pth')
            bin_score_modality_model = SimpleUNet(in_channels=len(config['clients'][str(client)])).to(device)
            checkpoint = torch.load(bin_score_mod_path, map_location=device)
            bin_score_modality_model.load_state_dict(checkpoint['model_state_dict'])
            bin_score_modality_model.eval()
            pretrained_models['bin_score_modality_model'] = bin_score_modality_model
        
        all_pretrained_models[client] = pretrained_models
        print(f"Loaded all pretrained models for client {client}")
    
    # Initialize local modality models and alignment losses
    local_modality_models = {}
    local_alignment_losses = {}
    prev_local_unets = {} 

    print("Modality optimizers created for all clients.")
    for client in args.clients:
        local_modality_models[client] = SimpleUNet(in_channels=len(config['clients'][str(client)])).to(device)
        prev_local_unets[client] = None
        
        # Create alignment loss based on type
        if args.alignment_type == 'statistics':
            local_alignment_losses[client] = FeatureStatisticsAlignmentLoss(
                num_channels=3
            ).to(device)
        elif args.alignment_type == 'higher_order':
            local_alignment_losses[client] = HigherOrderStatisticsAlignmentLoss(
                num_channels=3,
                use_skewness=True,
                use_kurtosis=True,
                use_percentiles=True,
                use_energy=True,
                use_correlation=True
            ).to(device)
        elif args.alignment_type == 'histogram':
            local_alignment_losses[client] = HistogramPrototypeAlignmentLoss(
                num_channels=3,
                num_bins=32
            ).to(device)
        elif args.alignment_type == 'spectral':
            local_alignment_losses[client] = SpectralPrototypeAlignmentLoss(
                num_channels=3,
                num_freq_bands=args.num_of_freq_bands
            ).to(device)
        elif args.alignment_type == 'all_stats':
            class CombinedStatisticalPrototypes(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.basic_stats = FeatureStatisticsAlignmentLoss(num_channels=3)
                    self.higher_order = HigherOrderStatisticsAlignmentLoss(
                        num_channels=3,
                        use_skewness=True,
                        use_kurtosis=True,
                        use_percentiles=True,
                        use_energy=True,
                        use_correlation=True
                    )
                    self.histogram = HistogramPrototypeAlignmentLoss(num_channels=3, num_bins=32)
                    self.spectral = SpectralPrototypeAlignmentLoss(num_channels=3, num_freq_bands=8)
                
                def forward(self, features):
                    loss1 = self.basic_stats(features)
                    loss2 = 0.5 * self.higher_order(features)
                    loss3 = 0.3 * self.histogram(features)
                    loss4 = 0.2 * self.spectral(features)
                    return loss1 + loss2 + loss3 + loss4
                
                def compute_local_statistics(self, features):
                    stats = {}
                    stats.update(self.basic_stats.compute_local_statistics(features))
                    stats.update(self.higher_order.compute_local_statistics(features))
                    stats.update(self.histogram.compute_local_statistics(features))
                    stats.update(self.spectral.compute_local_statistics(features))
                    return stats
                
                def update_global_statistics(self, global_stats):
                    self.basic_stats.update_global_statistics(global_stats)
                    self.higher_order.update_global_statistics(global_stats)
                    self.histogram.update_global_statistics(global_stats)
                    self.spectral.update_global_statistics(global_stats)
            
            local_alignment_losses[client] = CombinedStatisticalPrototypes().to(device)
    
    # Initialize map alignment losses (1 channel)
    local_map_alignment_losses = {}
    if args.map_alignment_weight > 0:
        print(f"Creating map alignment losses with type: {args.alignment_type}")
        for client in args.clients:
            # Create alignment loss based on type (1 channel for map_att)
            if args.alignment_type == 'statistics':
                local_map_alignment_losses[client] = FeatureStatisticsAlignmentLoss(
                    num_channels=1
                ).to(device)
            elif args.alignment_type == 'higher_order':
                local_map_alignment_losses[client] = HigherOrderStatisticsAlignmentLoss(
                    num_channels=1,
                    use_skewness=True,
                    use_kurtosis=True,
                    use_percentiles=True,
                    use_energy=True,
                    use_correlation=True
                ).to(device)
            elif args.alignment_type == 'histogram':
                local_map_alignment_losses[client] = HistogramPrototypeAlignmentLoss(
                    num_channels=4,
                    num_bins=32
                ).to(device)
            elif args.alignment_type == 'spectral':
                local_map_alignment_losses[client] = SpectralPrototypeAlignmentLoss(
                    num_channels=4,
                    num_freq_bands=args.num_of_freq_bands
                ).to(device)
            elif args.alignment_type == 'all_stats':
                class CombinedMapStatisticalPrototypes(nn.Module):
                    def __init__(self):
                        super().__init__()
                        self.basic_stats = FeatureStatisticsAlignmentLoss(num_channels=1)
                        self.higher_order = HigherOrderStatisticsAlignmentLoss(
                            num_channels=1,
                            use_skewness=True,
                            use_kurtosis=True,
                            use_percentiles=True,
                            use_energy=True,
                            use_correlation=True
                        )
                        self.histogram = HistogramPrototypeAlignmentLoss(num_channels=1, num_bins=32)
                        self.spectral = SpectralPrototypeAlignmentLoss(num_channels=1, num_freq_bands=8)
                    
                    def forward(self, features):
                        loss1 = self.basic_stats(features)
                        loss2 = 0.5 * self.higher_order(features)
                        loss3 = 0.3 * self.histogram(features)
                        loss4 = 0.2 * self.spectral(features)
                        return loss1 + loss2 + loss3 + loss4
                    
                    def compute_local_statistics(self, features):
                        stats = {}
                        stats.update(self.basic_stats.compute_local_statistics(features))
                        stats.update(self.higher_order.compute_local_statistics(features))
                        stats.update(self.histogram.compute_local_statistics(features))
                        stats.update(self.spectral.compute_local_statistics(features))
                        return stats
                    
                    def update_global_statistics(self, global_stats):
                        self.basic_stats.update_global_statistics(global_stats)
                        self.higher_order.update_global_statistics(global_stats)
                        self.histogram.update_global_statistics(global_stats)
                        self.spectral.update_global_statistics(global_stats)
                
                local_map_alignment_losses[client] = CombinedMapStatisticalPrototypes().to(device)
    
    # Setup dataloaders
    train_dataloaders = {}
    val_dataloaders = {}
    num_train_samples_clients = []
    modality_optimizers = {}
    for client_id in args.clients:
        modality_optimizers[client_id] = optim.Adam(
            local_modality_models[client_id].parameters(), 
            lr=args.learning_rate, 
            weight_decay=1e-5
        )
    loader_kwargs = {
        'num_workers': args.num_workers,
        'pin_memory': True,
    }
    if args.num_workers > 0:
        loader_kwargs['persistent_workers'] = True
        loader_kwargs['prefetch_factor'] = 2

    for client in args.clients:
        train_df = df[(df['client_id'] == client) & (df['split'] == "train")].reset_index(drop=True)
        num_train_samples_clients.append(len(train_df))
        train_dataset = ImageDataset(train_df, args.img_size, config, mode='train', client_id=client)
        train_dataloaders[client] = DataLoader(train_dataset, batch_size=args.batch_size, 
                                              shuffle=True, **loader_kwargs)
        
        val_df = df[(df['client_id'] == client) & (df['split'] == "val")].reset_index(drop=True)
        val_dataset = ImageDataset(val_df, args.img_size, config, mode='val', client_id=client)
        val_dataloaders[client] = DataLoader(val_dataset, batch_size=args.batch_size, 
                                            shuffle=False, **loader_kwargs)
    
    best_local_val_loss = {client: float('inf') for client in args.clients}
    start_round = 0
    
    # Early stopping setup
    patience = 25
    patience_counter = 0
    best_avg_val_loss = float('inf')
    
    # Load checkpoint if specified
    if args.checkpoint is not None:
        start_round = args.checkpoint + 1
        checkpoint_path = model_save_path + f"_scoring_round_{args.checkpoint}_client_global.pth"
        
        if os.path.exists(checkpoint_path):
            print(f"\n{'='*60}")
            print(f"Loading checkpoint from round {args.checkpoint}")
            print(f"{'='*60}\n")
            
            checkpoint = torch.load(checkpoint_path, map_location=device)
            global_agg_net.load_state_dict(checkpoint['model_state_dict'])
            
            if 'local_best' in checkpoint:
                for idx, client in enumerate(args.clients):
                    if idx < len(checkpoint['local_best']):
                        best_local_val_loss[client] = checkpoint['local_best'][idx]
            
            print(f"Loaded global aggregation network from round {args.checkpoint}")
            
            # Load modality models
            for client in args.clients:
                modality_checkpoint_path = model_save_path + f"_modality_round_{args.checkpoint}_client_{client}.pth"
                if os.path.exists(modality_checkpoint_path):
                    modality_checkpoint = torch.load(modality_checkpoint_path, map_location=device)
                    local_modality_models[client].load_state_dict(modality_checkpoint['model_state_dict'])
                    
                    if 'alignment_state_dict' in modality_checkpoint:
                        local_alignment_losses[client].load_state_dict(modality_checkpoint['alignment_state_dict'])
                    
                    if 'map_alignment_state_dict' in modality_checkpoint and args.map_alignment_weight > 0:
                        local_map_alignment_losses[client].load_state_dict(modality_checkpoint['map_alignment_state_dict'])
                    
                    if 'val_loss' in modality_checkpoint:
                        if modality_checkpoint['val_loss'] < best_local_val_loss[client]:
                            best_local_val_loss[client] = modality_checkpoint['val_loss']
                else:
                    print(f"Warning: No modality checkpoint found for client {client} at round {args.checkpoint}")
            
            print(f"{'='*60}\n")
        else:
            print(f"Warning: Checkpoint {checkpoint_path} not found, starting from scratch")
            raise FileNotFoundError(f"Checkpoint {checkpoint_path} not found")
    
    # Training loop
    weights = np.array(num_train_samples_clients) / np.sum(num_train_samples_clients)
    weights_tensor = torch.tensor(weights, dtype=torch.float32).to(device)

    for round in range(start_round, args.num_rounds):
        if round % 5 == 0:
            torch.cuda.empty_cache()
        
        print(f"\n{'='*60}\nRound {round+1}/{args.num_rounds}\n{'='*60}")
        
        # Client training
        client_agg_nets = [copy.deepcopy(global_agg_net) for _ in range(len(args.clients))]
        client_statistics = []
        client_map_statistics = []
        val_losses = []  # Track validation losses for early stopping
        
        for i, client_id in enumerate(args.clients):
            print(f"\nTraining Client {client_id}...")
            
            # Create trainer
            trainer = AggNetWithStatisticalPrototypes(
                args=args,
                model=client_agg_nets[i],
                modality_model=local_modality_models[client_id],
                alignment_loss=local_alignment_losses[client_id],
                train_loader=train_dataloaders[client_id],
                val_loader=val_dataloaders[client_id],
                client_id=client_id,
                pretrained_models=all_pretrained_models[client_id],
                alignment_weight=args.alignment_weight,
                modality_optimizer=modality_optimizers[client_id],
                map_alignment_loss=local_map_alignment_losses.get(client_id, None),
                map_alignment_weight=args.map_alignment_weight,
                fed_method=args.fed_method,
                global_agg_net = global_agg_net,
                prev_unet = prev_local_unets[client_id],
                mu = args.mu,
                temperature = args.temperature
            )
            #validate before training
            val_loss = trainer.val()
            print(f"Client {client_id} - Pre-Train Val Loss: {val_loss:.4f}")
            if val_loss < best_local_val_loss[client_id]:
                best_local_val_loss[client_id] = val_loss
                print(f"New best model for client {client_id} before training!")
                helper_save_model(round-1, client_id, client_agg_nets[i], 
                                model_save_path + "_personalized_scoring", val_loss, local_best=True)
                helper_save_model(round-1, client_id, local_modality_models[client_id],
                                model_save_path + "_personalized_modality", val_loss, local_best=True)
            # Train
            result = trainer.train(epoch=round, collect_stats=True)
            client_statistics.append(result['statistics'])
            client_map_statistics.append(result['map_statistics'])
            #copy current unet to prev unet for next round
            prev_local_unets[client_id] = copy.deepcopy(client_agg_nets[i])
            prev_local_unets[client_id].eval()
            
            print(f"Client {client_id} - Agg Loss: {result['agg_loss']:.4f}, "
                  f"Align Loss: {result['alignment_loss']:.4f}, "
                  f"Map Align Loss: {result['map_alignment_loss']:.4f}")
            
            # Validate
            val_loss = trainer.val()
            val_losses.append(val_loss)
            print(f"Client {client_id} - Val Loss: {val_loss:.4f}")
            
            # Save best local models
            if val_loss < best_local_val_loss[client_id]:
                best_local_val_loss[client_id] = val_loss
                print(f"New best model for client {client_id}!")
                helper_save_model(round, client_id, client_agg_nets[i], 
                                model_save_path + "_personalized_scoring", val_loss, local_best=True)
                helper_save_model(round, client_id, local_modality_models[client_id],
                                model_save_path + "_personalized_modality", val_loss, local_best=True)
        
        # Calculate average validation loss for early stopping
        avg_val_loss = np.mean(val_losses)
        print(f"\nAverage Validation Loss across clients: {avg_val_loss:.4f}")
        
        # Early stopping check
        if avg_val_loss < best_avg_val_loss:
            best_avg_val_loss = avg_val_loss
            patience_counter = 0
            print(f"New best average validation loss! Counter reset.")
        else:
            patience_counter += 1
            print(f"No improvement. Patience counter: {patience_counter}/{patience}")
            
            if patience_counter >= patience:
                print(f"\n{'='*60}")
                print(f"Early stopping triggered after {patience} rounds without improvement!")
                print(f"Best average validation loss: {best_avg_val_loss:.4f}")
                print(f"{'='*60}")
                break

        # Server aggregation
        print("\nAggregating models...")
        global_agg_net = aggregate_models(
            client_agg_nets,
            weights_tensor,
            device,
            global_model_prev=global_agg_net_prev,
            current_round=round + 1,
            cosine_agg_method=args.cosine_agg_method,
            cosine_start_round=args.cosine_start_round
        )

        # Save current global model as previous for next round
        global_agg_net_prev = copy.deepcopy(global_agg_net)
        global_agg_net_prev.eval()
        
        # Aggregate statistical prototypes
        global_stats = aggregate_statistics(client_statistics, weights)
        if global_stats is not None:
            print(f"Aggregated {len(global_stats)} statistical prototypes")
            for client_id in args.clients:
                local_alignment_losses[client_id].update_global_statistics(global_stats)
        
        # Aggregate map statistical prototypes
        global_map_stats = aggregate_statistics(client_map_statistics, weights)
        if global_map_stats is not None and args.map_alignment_weight > 0:
            print(f"Aggregated {len(global_map_stats)} map statistical prototypes")
            for client_id in args.clients:
                local_map_alignment_losses[client_id].update_global_statistics(global_map_stats)
        # evaluate global model on all clients
        print("\nEvaluating global aggregation network on all clients...")
        global_val_losses = []
        for client_id in args.clients:
            evaluator = AggNetWithStatisticalPrototypes(
                args=args,
                model=global_agg_net,
                modality_model=local_modality_models[client_id],
                alignment_loss=local_alignment_losses[client_id],
                train_loader=None,
                val_loader=val_dataloaders[client_id],
                client_id=client_id,
                pretrained_models=all_pretrained_models[client_id],
                alignment_weight=args.alignment_weight,
                modality_optimizer=None,
                map_alignment_loss=local_map_alignment_losses.get(client_id, None),
                map_alignment_weight=args.map_alignment_weight,
                fed_method=args.fed_method,
                global_agg_net = global_agg_net,
                prev_unet = prev_local_unets[client_id],
                mu = args.mu,
                temperature = args.temperature
            )
            val_loss = evaluator.val()
            global_val_losses.append(val_loss)
            print(f"Client {client_id} - Global Model Val Loss: {val_loss:.4f}")
        avg_global_val_loss = np.mean(global_val_losses)
        print(f"Average Global Model Validation Loss: {avg_global_val_loss:.4f}")
        if avg_global_val_loss < global_best_val_loss:
            global_best_val_loss = avg_global_val_loss
            print(f"New best global model based on average validation loss!")
            torch.save({
                'round': round,
                'model_state_dict': global_agg_net.state_dict(),
                'val_loss': avg_global_val_loss,
                'local_best': [best_local_val_loss[c] for c in args.clients]
            }, model_save_path + f"_global_best_scoring_model.pth")
            for client_id in args.clients:
                modality_save_path = model_save_path + f"_global_best_modality_client_{client_id}.pth"
                torch.save({
                    'round': round,
                    'model_state_dict': local_modality_models[client_id].state_dict(),
                    'alignment_state_dict': local_alignment_losses[client_id].state_dict(),
                    'val_loss': best_local_val_loss[client_id],
                }, modality_save_path)
        # Save checkpoint every 5 rounds
        if (round + 1) % 5 == 0:
            torch.save({
                'round': round,
                'model_state_dict': global_agg_net.state_dict(),
                'val_loss': avg_val_loss,
                'local_best': [best_local_val_loss[c] for c in args.clients]
            }, model_save_path + f"_scoring_round_{round}_client_global.pth")
            print(f"Saved global scoring model checkpoint at round {round}")
            #save previous local unets
            torch.save({
                'round': round,
                'prev_local_unets': {client_id: prev_local_unets[client_id].state_dict() for client_id in args.clients}
            }, model_save_path + f"_prev_local_unets_round_{round}.pth")
            print(f"Saved previous local UNet models at round {round}")
            
            for client_id in args.clients:
                modality_save_path = model_save_path + f"_modality_round_{round}_client_{client_id}.pth"
                save_dict = {
                    'round': round,
                    'model_state_dict': local_modality_models[client_id].state_dict(),
                    'alignment_state_dict': local_alignment_losses[client_id].state_dict(),
                    'val_loss': best_local_val_loss[client_id],
                }
                if args.map_alignment_weight > 0:
                    save_dict['map_alignment_state_dict'] = local_map_alignment_losses[client_id].state_dict()
                torch.save(save_dict, modality_save_path)
                print(f"Saved modality model and alignment loss for client {client_id} at round {round}")
            
            # Remove old checkpoints (keep only last 2)
            if round >= 10:
                old_round = round - 10
                old_unet_path = model_save_path + f"_scoring_round_{old_round}_client_global.pth"
                if os.path.exists(old_unet_path):
                    os.remove(old_unet_path)
                    print(f"Removed old scoring checkpoint from round {old_round}")
                
                for client_id in args.clients:
                    old_modality_path = model_save_path + f"_modality_round_{old_round}_client_{client_id}.pth"
                    if os.path.exists(old_modality_path):
                        os.remove(old_modality_path)
                        print(f"Removed old modality checkpoint for client {client_id} from round {old_round}")
                # Remove old prev unet checkpoints
                old_prev_unet_path = model_save_path + f"_prev_local_unets_round_{old_round}.pth"
                if os.path.exists(old_prev_unet_path):
                    os.remove(old_prev_unet_path)
                    print(f"Removed old previous local UNets checkpoint from round {old_round}")

    
    print("\nTraining completed!")
    print(f"Best average validation loss: {best_avg_val_loss:.4f}")
    for client_id in args.clients:
        print(f"Client {client_id} best val loss: {best_local_val_loss[client_id]:.4f}")
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Total training time: {elapsed_time/60:.2f} minutes")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    # Basic args
    parser.add_argument('--clients', type=str, default="4,6,13,20")
    parser.add_argument('--num_rounds', type=int, default=100)
    parser.add_argument('--epochs', type=int, default=5, help="Local epochs per round")
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--img_size', type=int, default=224)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--csv_path', type=str, required=True)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--save_dir', type=str, default='./checkpoints/agg_net_stat_proto/')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--checkpoint', type=int, default=None, 
                       help='Round number to resume from (e.g., 50 to resume from round 50 checkpoint)')
    
    # Task settings
    parser.add_argument('--task', type=str, default='binary', choices=['binary', 'multiclass'])
    parser.add_argument('--num_classes', type=int, default=1)
    parser.add_argument('--loss_weight', type=float, nargs='+', default=None)
    
    # Alignment settings
    parser.add_argument('--alignment_type', type=str, default='higher_order',
                       choices=['statistics', 'higher_order', 'histogram', 'spectral', 'all_stats'])
    parser.add_argument('--alignment_weight', type=float, default=0.1)
    parser.add_argument('--map_alignment_weight', type=float, default=0.1,
                       help='Weight for map attention alignment loss (set to 0 to disable)')
    
    # Cosine similarity aggregation options
    parser.add_argument('--cosine_agg_method', type=str, default=None,
                       choices=[None, 'layer_wise', 'filter_wise_client', 'filter_wise_global'],
                       help='Cosine similarity-based aggregation method (None for standard FedAvg)')
    parser.add_argument('--cosine_start_round', type=int, default=11,
                       help='Round to start using cosine similarity aggregation (default: 11)')
    
    # Pretrained model base directories (will auto-construct paths)
    parser.add_argument('--bin_pretrained_dir', type=str, default=None,
                       help='Base directory for binary classifier models')
    parser.add_argument('--bin_modality_pretrained_dir', type=str, default=None,
                       help='Base directory for binary modality models')
    parser.add_argument('--multi_pretrained_dir', type=str, default=None,
                       help='Base directory for multiclass classifier models')
    parser.add_argument('--multi_modality_pretrained_dir', type=str, default=None,
                       help='Base directory for multiclass modality models')
    parser.add_argument('--bin_score_pretrained_dir', type=str, default=None,
                       help='Base directory for binary score models')
    parser.add_argument('--bin_score_modality_pretrained_dir', type=str, default=None,
                       help='Base directory for binary score modality models')
    parser.add_argument('--fed_method', type=str, default='fedavg',
                       choices=['fedavg', 'fedprox', 'moon'],
                       help='Federated learning method: fedavg, fedprox, or moon')
    parser.add_argument('--mu', type=float, default=0.01,
                       help='Proximal term coefficient for FedProx (default: 0.01)')
    parser.add_argument('--temperature', type=float, default=0.5,
                       help='Temperature for MOON contrastive loss (default: 0.5)')
    parser.add_argument('--amp', action='store_true',
                       help='Enable mixed precision training (CUDA only)')
    parser.add_argument('--amp_dtype', type=str, default='float16', choices=['float16', 'bfloat16'],
                       help='AMP dtype to use when --amp is set')
    parser.add_argument('--log_dir', type=str, default='nohups',
                       help='Directory to write logs')
    parser.add_argument('--log_file', type=str, default=None,
                       help='Optional explicit log file path')
    
    parser.add_argument('--num_of_freq_bands', type=int, default=8,
                       help='Number of frequency bands for spectral prototypes (default: 8)')
    
    # CAM quality improvement flags
    parser.add_argument('--spatial_normalize', action='store_true',
                       help='Normalize over spatial dim (h*w) instead of classes in Res_Scoring. '
                            'Prevents one class from suppressing the other. Requires agg-net retrain.')
    parser.add_argument('--soft_gating_alpha', type=float, default=0.0,
                       help='Leakage factor for soft binary gating (0=hard gating, 0.3=recommended). '
                            'Keeps alpha fraction of signal even when binary gate is off. Requires agg-net retrain.')
    parser.add_argument('--per_class_agreement', action='store_true',
                       help='Apply agreement loss to each class independently instead of only max class. '
                            'Ensures both core and edema get gradient supervision. Requires agg-net retrain.')

    args = parser.parse_args()

    setup_tee_logging(args.log_dir, args.log_file)
    
    # Parse clients
    args.clients = [int(c) for c in args.clients.split(',')]
    
    federated_train_agg_net_statistical_prototypes(args)
