"""
Federated Training with Statistical Prototype Alignment + FedProx + MOON

This extends train_federated_with_dual_alignment.py with additional federated methods:
- FedAvg (default)
- FedProx: Adds proximal term to keep local models close to global model
- MOON: Model-contrastive federated learning with representation similarity

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
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
from models import Res18_Classifier, SimpleUNet
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


def helper_train_with_alignment(unet_model, modality_model, alignment_loss, train_loader, 
                                  unet_optimizer, modality_optimizer, criterion, device, 
                                  alignment_weight=0.1, collect_stats=False,
                                  l1_alignment_loss=None, l1_alignment_weight=0.1,
                                  fed_method='fedavg', global_unet=None, prev_unet=None,
                                  mu=0.01, temperature=0.5, use_amp=False, amp_dtype=torch.float16):
    """
    Train with modality alignment loss and optional FedProx/MOON
    
    Args:
        fed_method: 'fedavg', 'fedprox', or 'moon'
        global_unet: Global model (needed for FedProx and MOON)
        prev_unet: Previous local model (needed for MOON)
        mu: Proximal term coefficient for FedProx (default: 0.01)
        temperature: Temperature for MOON contrastive loss (default: 0.5)
    """
    stats_to_aggregate = []
    l1_stats_to_aggregate = []
    amp_enabled = use_amp and str(device).startswith("cuda")
    scaler = GradScaler(enabled=amp_enabled)
    
    for epoch in range(5):
        unet_model.train()
        modality_model.train()
        train_bar = train_loader
        total_loss, total_num = 0.0, 0
        total_task_loss, total_align_loss, total_l1_align_loss = 0.0, 0.0, 0.0
        total_prox_loss, total_moon_loss = 0.0, 0.0
        
        for case_batch, label_batch in train_bar:
            case_batch = case_batch.to(device, non_blocking=True)
            label_batch = label_batch.to(device, non_blocking=True)
            
            # Zero gradients
            unet_optimizer.zero_grad(set_to_none=True)
            modality_optimizer.zero_grad(set_to_none=True)
            
            with autocast(enabled=amp_enabled, dtype=amp_dtype):
                # Forward through modality model
                modality_features = modality_model(case_batch)
                
                # Task loss (classification) - get l1 features if needed
                if l1_alignment_loss is not None:
                    logits_collect, _, l1_features = unet_model(modality_features, return_maps=False, return_l1_features=True)
                else:
                    logits_collect, _ = unet_model(modality_features)
                task_loss = 0
                ic_weight = [0.25, 0.5, 0.75, 1.0]
                
                for idx, logits in enumerate(logits_collect):
                    loss_val = criterion(logits, label_batch.float())
                    task_loss += ic_weight[idx] * loss_val
                
                # Alignment loss (encourages similar features across clients)
                if isinstance(alignment_loss, CombinedModalityAlignmentLoss):
                    align_loss, _ = alignment_loss(modality_features)
                else:
                    align_loss = alignment_loss(modality_features)
                
                # L1 alignment loss (if enabled)
                l1_align_loss = 0
                if l1_alignment_loss is not None:
                    if isinstance(l1_alignment_loss, CombinedModalityAlignmentLoss):
                        l1_align_loss, _ = l1_alignment_loss(l1_features)
                    else:
                        l1_align_loss = l1_alignment_loss(l1_features)
                
                # Combined loss
                loss = task_loss + alignment_weight * align_loss
                if l1_alignment_loss is not None:
                    loss = loss + l1_alignment_weight * l1_align_loss
                
                # === FedProx: Add proximal term ===
                prox_loss = 0
                if fed_method == 'fedprox' and global_unet is not None:
                    for w, w_g in zip(unet_model.parameters(), global_unet.parameters()):
                        prox_loss += (w - w_g).norm(2)
                    prox_loss = (mu / 2) * prox_loss
                    loss = loss + prox_loss
                
                # === MOON: Add contrastive loss ===
                moon_loss = 0
                if fed_method == 'moon' and global_unet is not None:
                    # Get representations from current, global, and previous models
                    with torch.no_grad():
                        global_unet.eval()
                        global_modality_features = modality_features.detach()
                        global_logits, _, global_l1 = global_unet(global_modality_features, return_maps=False, return_l1_features=True)
                        
                        if prev_unet is not None:
                            prev_unet.eval()
                            prev_logits, _, prev_l1 = prev_unet(global_modality_features, return_maps=False, return_l1_features=True)
                    
                    # Use l1 features as representations (64 channels, spatially pooled)
                    local_rep = F.adaptive_avg_pool2d(l1_features, (1, 1)).flatten(1)
                    global_rep = F.adaptive_avg_pool2d(global_l1, (1, 1)).flatten(1)
                    
                    # Normalize representations
                    local_rep = F.normalize(local_rep, dim=1)
                    global_rep = F.normalize(global_rep, dim=1)
                    
                    # Positive similarity (with global model)
                    pos_sim = torch.exp(torch.sum(local_rep * global_rep, dim=1) / temperature)
                    
                    # Negative similarity (with previous local model if available)
                    if prev_unet is not None:
                        prev_rep = F.adaptive_avg_pool2d(prev_l1, (1, 1)).flatten(1)
                        prev_rep = F.normalize(prev_rep, dim=1)
                        neg_sim = torch.exp(torch.sum(local_rep * prev_rep, dim=1) / temperature)
                        moon_loss = -torch.log(pos_sim / (pos_sim + neg_sim)).mean()
                    else:
                        # If no previous model, just maximize similarity with global
                        moon_loss = -torch.log(pos_sim / (pos_sim + 1e-8)).mean()
                    
                    loss = loss + moon_loss
            
            scaler.scale(loss).backward()
            
            # Update both models
            scaler.step(unet_optimizer)
            scaler.step(modality_optimizer)
            scaler.update()
            
            total_num += case_batch.size(0)
            total_loss += loss.item() * case_batch.size(0)
            total_task_loss += task_loss.item() * case_batch.size(0)
            total_align_loss += align_loss.item() * case_batch.size(0)
            if l1_alignment_loss is not None:
                total_l1_align_loss += l1_align_loss.item() * case_batch.size(0)
            if fed_method == 'fedprox':
                total_prox_loss += prox_loss.item() * case_batch.size(0)
            if fed_method == 'moon':
                total_moon_loss += moon_loss.item() * case_batch.size(0)
            
            # Collect statistics for aggregation
            if collect_stats:
                with torch.no_grad():
                    if hasattr(alignment_loss, 'compute_local_statistics'):
                        local_stats = alignment_loss.compute_local_statistics(modality_features)
                        if local_stats is not None:
                            stats_to_aggregate.append(local_stats)
                    
                    if l1_alignment_loss is not None and hasattr(l1_alignment_loss, 'compute_local_statistics'):
                        l1_local_stats = l1_alignment_loss.compute_local_statistics(l1_features)
                        if l1_local_stats is not None:
                            l1_stats_to_aggregate.append(l1_local_stats)
    
    avg_stats = None
    if stats_to_aggregate:
        avg_stats = {}
        stat_keys = stats_to_aggregate[0].keys()
        for key in stat_keys:
            values = [s[key] for s in stats_to_aggregate if key in s]
            if values:
                avg_stats[key] = torch.stack(values).mean(dim=0)
    
    avg_l1_stats = None
    if l1_stats_to_aggregate:
        avg_l1_stats = {}
        stat_keys = l1_stats_to_aggregate[0].keys()
        for key in stat_keys:
            values = [s[key] for s in l1_stats_to_aggregate if key in s]
            if values:
                avg_l1_stats[key] = torch.stack(values).mean(dim=0)
    
    result = {
        'task_loss': total_task_loss / total_num,
        'alignment_loss': total_align_loss / total_num,
        'l1_alignment_loss': total_l1_align_loss / total_num,
        'statistics': avg_stats,
        'l1_statistics': avg_l1_stats
    }
    
    if fed_method == 'fedprox':
        result['prox_loss'] = total_prox_loss / total_num
    if fed_method == 'moon':
        result['moon_loss'] = total_moon_loss / total_num
    
    return result


def helper_validate(unet_model, modality_model, val_loader, device, criterion, use_amp=False, amp_dtype=torch.float16):
    unet_model.eval()
    modality_model.eval()
    amp_enabled = use_amp and str(device).startswith("cuda")
    
    pred_results = []
    val_labels = []
    total_loss, total_num = 0.0, 0
    
    with torch.no_grad():
        for case_batch, label_batch in val_loader:
            case_batch = case_batch.to(device, non_blocking=True)
            label_batch = label_batch.to(device, non_blocking=True)
            
            with autocast(enabled=amp_enabled, dtype=amp_dtype):
                modality_features = modality_model(case_batch)
                logits_collect, _ = unet_model(modality_features)
                
                pred = torch.sigmoid(logits_collect[-1])
                loss = criterion(logits_collect[-1], label_batch.float())
            
            total_num += case_batch.size(0)
            total_loss += loss.item() * case_batch.size(0)
            pred_results.append(pred.cpu())
            val_labels.append(label_batch.cpu())

    pred_results = torch.cat(pred_results, dim=0).numpy()
    val_labels = torch.cat(val_labels, dim=0).numpy()
    val_acc, val_auc = evaluate(val_labels, pred_results)
    return val_acc


def evaluate(labels, predictions, threshold=0.5):
    predictions_binary = (predictions >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, predictions_binary).ravel()
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    
    try:
        auc = roc_auc_score(labels, predictions)
    except:
        auc = 0.0
    
    return accuracy, auc


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
            - 'layer_wise': Layer-wise cosine similarity with separate weights per layer
            - 'filter_wise_client': Filter-wise cosine per client, then FedAvg
            - 'filter_wise_global': Filter-wise cosine across all clients, normalize per filter
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
        print(f"  Current Round: {current_round}, Using Cosine Similarity Aggregation Method: {cosine_agg_method}")
        return _aggregate_layer_wise_cosine(client_models, weights, global_model, global_model_prev, device)
    elif cosine_agg_method == 'filter_wise_client':
        return _aggregate_filter_wise_client_cosine(client_models, weights, global_model, global_model_prev, device)
    elif cosine_agg_method == 'filter_wise_global':
        return _aggregate_filter_wise_global_cosine(client_models, weights, global_model, global_model_prev, device)
    else:
        raise ValueError(f"Unknown cosine_agg_method: {cosine_agg_method}")


def _aggregate_layer_wise_cosine(client_models, data_weights, global_model, global_model_prev, device):
    """
    Method 1: Layer-wise Cosine Similarity
    Each layer gets its own cosine-based weight for each client.
    Don't take mean - update each layer separately with separate weights.
    """
    print(f"  Using Layer-wise Cosine Similarity Aggregation")
    
    # Zero out global model
    for global_param in global_model.parameters():
        global_param.data = torch.zeros_like(global_param.data)
    
    # Get parameter names for classification head detection
    param_names = [name for name, _ in global_model.named_parameters()]
    
    # Iterate through each layer
    for param_idx, (global_param, prev_param) in enumerate(zip(
        global_model.parameters(), 
        global_model_prev.parameters()
    )):
        param_name = param_names[param_idx]
        
        # Skip classification head
        if 'classifier' in param_name.lower() or 'fc' in param_name.lower() or 'head' in param_name.lower():
            # Use standard FedAvg for classification head
            for client_model, weight in zip(client_models, data_weights):
                client_param = list(client_model.parameters())[param_idx]
                global_param.data += weight * client_param.data
            continue
        
        # Compute cosine similarity for this layer for each client
        layer_cosine_weights = []
        for client_model in client_models:
            client_param = list(client_model.parameters())[param_idx]
            
            # Flatten parameters for cosine similarity
            client_flat = client_param.data.flatten()
            prev_flat = prev_param.data.flatten()
            
            # Cosine similarity
            cos_sim = torch.nn.functional.cosine_similarity(
                client_flat.unsqueeze(0), 
                prev_flat.unsqueeze(0), 
                dim=1
            ).item()
            
            # Clamp to [0, 1] - negative similarity means opposite direction
            cos_sim = max(cos_sim, 0.0)
            layer_cosine_weights.append(cos_sim)
        
        # Normalize cosine weights for this layer
        layer_cosine_weights = torch.tensor(layer_cosine_weights, device=device)
        if layer_cosine_weights.sum() > 0:
            layer_cosine_weights = layer_cosine_weights / layer_cosine_weights.sum()
        else:
            # Fallback to data weights if all similarities are 0
            layer_cosine_weights = torch.tensor(data_weights, device=device)
        
        # Aggregate this layer with layer-specific weights
        for client_idx, client_model in enumerate(client_models):
            client_param = list(client_model.parameters())[param_idx]
            global_param.data += layer_cosine_weights[client_idx] * client_param.data
    
    return global_model


def _aggregate_filter_wise_client_cosine(client_models, data_weights, global_model, global_model_prev, device):
    """
    Method 2: Filter-wise Cosine per Client
    1. Calculate cosine similarity for each filter
    2. Normalize them for each client separately
    3. Multiply them with each filter
    4. Then use current FedAvg logic to weight them
    """
    print(f"  Using Filter-wise Client Cosine Similarity Aggregation")
    
    # Zero out global model
    for global_param in global_model.parameters():
        global_param.data = torch.zeros_like(global_param.data)
    
    param_names = [name for name, _ in global_model.named_parameters()]
    
    # Process each client
    weighted_client_models = []
    for client_idx, client_model in enumerate(client_models):
        # Create a copy to store cosine-weighted parameters
        weighted_model_params = []
        
        for param_idx, (client_param, prev_param) in enumerate(zip(
            client_model.parameters(),
            global_model_prev.parameters()
        )):
            param_name = param_names[param_idx]
            
            # Skip classification head - no cosine weighting
            if 'classifier' in param_name.lower() or 'fc' in param_name.lower() or 'head' in param_name.lower():
                weighted_model_params.append(client_param.data.clone())
                continue
            
            # For conv/linear layers with multiple filters
            if len(client_param.shape) >= 2:  # Conv or Linear layer
                filter_cosine_weights = []
                
                # Compute cosine similarity for each filter
                if len(client_param.shape) == 4:  # Conv: [out_ch, in_ch, h, w]
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
                    
                    # Normalize per client
                    filter_cosine_weights = torch.tensor(filter_cosine_weights, device=device)
                    min_val = filter_cosine_weights.min()
                    max_val = filter_cosine_weights.max()
                    if max_val > min_val:
                        filter_cosine_weights = (filter_cosine_weights - min_val) / (max_val - min_val)
                    else:
                        filter_cosine_weights = torch.ones_like(filter_cosine_weights)
                    
                    # Multiply each filter with its cosine weight
                    weighted_param = client_param.data.clone()
                    for filter_idx in range(num_filters):
                        weighted_param[filter_idx] *= filter_cosine_weights[filter_idx]
                    
                    weighted_model_params.append(weighted_param)
                
                elif len(client_param.shape) == 2:  # Linear: [out, in]
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
                    
                    # Normalize per client
                    filter_cosine_weights = torch.tensor(filter_cosine_weights, device=device)
                    min_val = filter_cosine_weights.min()
                    max_val = filter_cosine_weights.max()
                    if max_val > min_val:
                        filter_cosine_weights = (filter_cosine_weights - min_val) / (max_val - min_val)
                    else:
                        filter_cosine_weights = torch.ones_like(filter_cosine_weights)
                    
                    # Multiply each filter with its cosine weight
                    weighted_param = client_param.data.clone()
                    for filter_idx in range(num_filters):
                        weighted_param[filter_idx] *= filter_cosine_weights[filter_idx]
                    
                    weighted_model_params.append(weighted_param)
                else:
                    # For other shapes, just copy
                    weighted_model_params.append(client_param.data.clone())
            else:
                # For 1D parameters (bias, BN params), just copy
                weighted_model_params.append(client_param.data.clone())
        
        # Now aggregate with standard FedAvg using data weights
        for param_idx, (global_param, weighted_param) in enumerate(zip(
            global_model.parameters(),
            weighted_model_params
        )):
            global_param.data += data_weights[client_idx] * weighted_param
    
    return global_model


def _aggregate_filter_wise_global_cosine(client_models, data_weights, global_model, global_model_prev, device):
    """
    Method 3: Filter-wise Cosine Global Normalization
    1. Calculate cosine similarity for each filter for each client
    2. For each filter, have N values (N clients)
    3. Normalize these N values per filter
    4. Multiply with their own filter weights
    5. Then use FedAvg
    """
    print(f"  Using Filter-wise Global Cosine Similarity Aggregation")
    
    # Zero out global model
    for global_param in global_model.parameters():
        global_param.data = torch.zeros_like(global_param.data)
    
    param_names = [name for name, _ in global_model.named_parameters()]
    num_clients = len(client_models)
    
    # Iterate through each parameter/layer
    for param_idx, (global_param, prev_param) in enumerate(zip(
        global_model.parameters(),
        global_model_prev.parameters()
    )):
        param_name = param_names[param_idx]
        
        # Skip classification head
        if 'classifier' in param_name.lower() or 'fc' in param_name.lower() or 'head' in param_name.lower():
            # Standard FedAvg for classification head
            for client_idx, client_model in enumerate(client_models):
                client_param = list(client_model.parameters())[param_idx]
                global_param.data += data_weights[client_idx] * client_param.data
            continue
        
        # Get all client parameters for this layer
        client_params = [list(client_model.parameters())[param_idx] for client_model in client_models]
        
        # For conv/linear layers with multiple filters
        if len(global_param.shape) >= 2:  # Conv or Linear layer
            if len(global_param.shape) == 4:  # Conv: [out_ch, in_ch, h, w]
                num_filters = global_param.shape[0]
                
                # For each filter, compute cosine similarities across all clients
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
                    
                    # Normalize across clients for this specific filter
                    filter_cosine_weights = torch.tensor(filter_cosine_weights, device=device)
                    min_val = filter_cosine_weights.min()
                    max_val = filter_cosine_weights.max()
                    if max_val > min_val:
                        filter_cosine_weights = (filter_cosine_weights - min_val) / (max_val - min_val)
                    else:
                        filter_cosine_weights = torch.ones_like(filter_cosine_weights)
                    
                    # Aggregate this filter across clients with normalized weights
                    for client_idx, client_param in enumerate(client_params):
                        global_param.data[filter_idx] += (
                            filter_cosine_weights[client_idx] * 
                            data_weights[client_idx] * 
                            client_param.data[filter_idx]
                        )
            
            elif len(global_param.shape) == 2:  # Linear: [out, in]
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
                    
                    # Normalize across clients for this specific filter
                    filter_cosine_weights = torch.tensor(filter_cosine_weights, device=device)
                    min_val = filter_cosine_weights.min()
                    max_val = filter_cosine_weights.max()
                    if max_val > min_val:
                        filter_cosine_weights = (filter_cosine_weights - min_val) / (max_val - min_val)
                    else:
                        filter_cosine_weights = torch.ones_like(filter_cosine_weights)
                    
                    # Aggregate this filter across clients
                    for client_idx, client_param in enumerate(client_params):
                        global_param.data[filter_idx] += (
                            filter_cosine_weights[client_idx] * 
                            data_weights[client_idx] * 
                            client_param.data[filter_idx]
                        )
            else:
                # For other shapes, standard FedAvg
                for client_idx, client_param in enumerate(client_params):
                    global_param.data += data_weights[client_idx] * client_param.data
        else:
            # For 1D parameters (bias, BN), standard FedAvg
            for client_idx, client_param in enumerate(client_params):
                global_param.data += data_weights[client_idx] * client_param.data
    
    return global_model


def aggregate_statistics(client_stats_list, weights):
    """Aggregate statistics from all clients with weighted average"""
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


def federated_train_with_statistical_prototypes(client_list, num_rounds, batch_size, img_size, num_workers, 
                                                  csv_path, starting_lr, save_dir, device, checkpoint=None,
                                                  alignment_type='statistics', alignment_weight=0.1,
                                                  cosine_agg_method=None, cosine_start_round=11,
                                                  l1_alignment_weight=0.1,
                                                  fed_method='fedavg', mu=0.01, temperature=0.5,
                                                  use_amp=False, amp_dtype=torch.float16):
    """
    Federated training with statistical prototype alignment + FedProx/MOON
    
    Args:
        fed_method: Federated learning method
            - 'fedavg': Standard FedAvg (default)
            - 'fedprox': FedProx with proximal term
            - 'moon': Model-contrastive federated learning
        mu: Proximal term coefficient for FedProx (default: 0.01)
        temperature: Temperature for MOON contrastive loss (default: 0.5)
    """
    
    temp_path = ""
    for client in client_list:
        temp_path = str(client) + "_" + temp_path
    model_save_path = save_dir + temp_path + "/"
    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path, exist_ok=True)
    
    # Save configuration
    config_dict = {
        'alignment_type': alignment_type,
        'alignment_weight': alignment_weight,
        'fed_method': fed_method,
        'mu': mu if fed_method == 'fedprox' else None,
        'temperature': temperature if fed_method == 'moon' else None,
        'clients': client_list,
        'num_rounds': num_rounds,
        'batch_size': batch_size,
        'learning_rate': starting_lr
    }
    with open(model_save_path + "config.json", 'w') as f:
        json.dump(config_dict, f, indent=2)
    
    df = pd.read_csv(csv_path)
    criterion = torch.nn.BCEWithLogitsLoss().to(device)
    
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
    align_weights = {
        "1": 0.1,
        "2": 0.1,
        "3": 0.1,
        "4": 0.1,
        "5": 0.1,
        "6": 0.1,
    }
    
    # Global UNet (aggregated)
    global_unet = Res18_Classifier(num_classes=1).to(device)
    global_unet_prev = None
    
    # Local modality models and alignment losses
    local_modality_models = {}
    local_alignment_losses = {}
    local_l1_alignment_losses = {}
    local_modality_optimizers = {}
    prev_local_unets = {}  # Store previous local models for MOON
    
    for client in client_list:
        local_modality_models[client] = SimpleUNet(in_channels=len(config['clients'][str(client)])).to(device)
        prev_local_unets[client] = None  # Initialize as None
        
        # Create alignment loss module based on type
        if alignment_type == 'statistics':
            local_alignment_losses[client] = FeatureStatisticsAlignmentLoss(num_channels=3).to(device)
        elif alignment_type == 'prototypes':
            local_alignment_losses[client] = PrototypeAlignmentLoss(num_prototypes=16, feature_dim=3).to(device)
        elif alignment_type == 'combined':
            local_alignment_losses[client] = CombinedModalityAlignmentLoss(
                use_statistics=True,
                use_prototypes=True,
                use_normalization=True
            ).to(device)
        elif alignment_type == 'higher_order':
            local_alignment_losses[client] = HigherOrderStatisticsAlignmentLoss(
                num_channels=3,
                use_skewness=True,
                use_kurtosis=True,
                use_percentiles=True,
                use_energy=True,
                use_correlation=True
            ).to(device)
        elif alignment_type == 'histogram':
            local_alignment_losses[client] = HistogramPrototypeAlignmentLoss(
                num_channels=3,
                num_bins=16
            ).to(device)
        elif alignment_type == 'spectral':
            local_alignment_losses[client] = SpectralPrototypeAlignmentLoss(
                num_channels=3,
                num_freq_bands=8
            ).to(device)
        elif alignment_type == 'all_stats':
            class CombinedStatisticalPrototypes(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.basic_stats = FeatureStatisticsAlignmentLoss(num_channels=3)
                    self.higher_order = HigherOrderStatisticsAlignmentLoss(
                        num_channels=3, use_skewness=True, use_kurtosis=True,
                        use_percentiles=True, use_energy=True, use_correlation=True
                    )
                    self.histogram = HistogramPrototypeAlignmentLoss(num_channels=3, num_bins=16)
                    self.spectral = SpectralPrototypeAlignmentLoss(num_channels=3, num_freq_bands=8)
                
                def forward(self, features):
                    loss1 = 0.1 * self.basic_stats(features)
                    loss2 = 0.2 * self.higher_order(features)
                    loss3 = 0.3 * self.histogram(features)
                    loss4 = 0.4 * self.spectral(features)
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
        else:
            raise ValueError(f"Unknown alignment_type: {alignment_type}")
        
        # L1 alignment (64 channels)
        if alignment_type == 'all_stats':
            class CombinedStatisticalPrototypesL1(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.basic_stats = FeatureStatisticsAlignmentLoss(num_channels=64)
                    self.higher_order = HigherOrderStatisticsAlignmentLoss(
                        num_channels=64, use_skewness=True, use_kurtosis=True,
                        use_percentiles=True, use_energy=True, use_correlation=True
                    )
                    self.histogram = HistogramPrototypeAlignmentLoss(num_channels=64, num_bins=16)
                    self.spectral = SpectralPrototypeAlignmentLoss(num_channels=64, num_freq_bands=8)
                
                def forward(self, features):
                    loss1 = 0.1 * self.basic_stats(features)
                    loss2 = 0.2 * self.higher_order(features)
                    loss3 = 0.3 * self.histogram(features)
                    loss4 = 0.4 * self.spectral(features)
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
            
        if alignment_type == 'statistics':
            local_l1_alignment_losses[client] = FeatureStatisticsAlignmentLoss(num_channels=64).to(device)
        elif alignment_type == 'prototypes':
            local_l1_alignment_losses[client] = PrototypeAlignmentLoss(num_prototypes=16, feature_dim=64).to(device)
        elif alignment_type == 'combined':
            local_l1_alignment_losses[client] = CombinedModalityAlignmentLoss(
                use_statistics=True,
                use_prototypes=True,
                use_normalization=True
            ).to(device)
        elif alignment_type == 'higher_order':
            local_l1_alignment_losses[client] = HigherOrderStatisticsAlignmentLoss(
                num_channels=64,
                use_skewness=True,
                use_kurtosis=True,
                use_percentiles=True,
                use_energy=True,
                use_correlation=True
            ).to(device)
        elif alignment_type == 'histogram':
            local_l1_alignment_losses[client] = HistogramPrototypeAlignmentLoss(
                num_channels=64,
                num_bins=16
            ).to(device)
        elif alignment_type == 'spectral':
            local_l1_alignment_losses[client] = SpectralPrototypeAlignmentLoss(
                num_channels=64,
                num_freq_bands=8
            ).to(device)
        elif alignment_type == 'all_stats':
            class CombinedStatisticalPrototypesL1(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.basic_stats = FeatureStatisticsAlignmentLoss(num_channels=64)
                    self.higher_order = HigherOrderStatisticsAlignmentLoss(
                        num_channels=64,
                        use_skewness=True,
                        use_kurtosis=True,
                        use_percentiles=True,
                        use_energy=True,
                        use_correlation=True
                    )
                    self.histogram = HistogramPrototypeAlignmentLoss(num_channels=64, num_bins=32)
                    self.spectral = SpectralPrototypeAlignmentLoss(num_channels=64, num_freq_bands=8)
                
                def forward(self, features):
                    loss1 = 0.1 * self.basic_stats(features)
                    loss2 = 0.2 * self.higher_order(features)
                    loss3 = 0.3 * self.histogram(features)
                    loss4 = 0.4 * self.spectral(features)
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
            
            local_l1_alignment_losses[client] = CombinedStatisticalPrototypesL1().to(device)
    
    
    # Create persistent optimizers for local modality models
    for client in client_list:
        local_modality_optimizers[client] = optim.Adam(
            local_modality_models[client].parameters(), 
            lr=starting_lr, 
            weight_decay=1e-5
        )
    
    # Setup dataloaders
    train_dataloaders = {}
    val_dataloaders = {}
    num_train_samples_clients = []
    loader_kwargs = {
        'num_workers': num_workers,
        'pin_memory': True,
    }
    if num_workers > 0:
        loader_kwargs['persistent_workers'] = True
        loader_kwargs['prefetch_factor'] = 2

    for client in client_list:
        train_df = df[(df['client_id'] == client) & (df['split'] == "train")].reset_index(drop=True)
        num_train_samples_clients.append(len(train_df))
        train_dataset = ImageDataset(train_df, img_size, config, mode='train', client_id=client)
        train_dataloaders[client] = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, **loader_kwargs)
        
        val_df = df[(df['client_id'] == client) & (df['split'] == "val")].reset_index(drop=True)
        val_dataset = ImageDataset(val_df, img_size, config, mode='val', client_id=client)
        val_dataloaders[client] = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, **loader_kwargs)
    
    best_acc = 0.0
    start_round = 0
    best_local_acc = {client: 0.0 for client in client_list}
    
    # Early stopping setup
    patience = 15
    patience_counter = 0
    best_avg_val_acc = 0.0
    
    # Load checkpoint if specified
    if checkpoint is not None:
        start_round = checkpoint + 1
        checkpoint_path = model_save_path + f"_unet_round_{checkpoint}_client_global.pth"
        if os.path.exists(checkpoint_path):
            checkpoint_dict = torch.load(checkpoint_path)
            global_unet.load_state_dict(checkpoint_dict['model_state_dict'])
            best_acc = checkpoint_dict.get('val_acc', 0.0)
            print(f"Loaded global UNet checkpoint from round {checkpoint}, Val Acc: {best_acc:.4f}")
            
            for client in client_list:
                modality_checkpoint_path = model_save_path + f"_modality_round_{checkpoint}_client_{client}.pth"
                if os.path.exists(modality_checkpoint_path):
                    modality_checkpoint = torch.load(modality_checkpoint_path)
                    local_modality_models[client].load_state_dict(modality_checkpoint['model_state_dict'])
                    if 'alignment_state_dict' in modality_checkpoint:
                        local_alignment_losses[client].load_state_dict(modality_checkpoint['alignment_state_dict'])
                    if 'l1_alignment_state_dict' in modality_checkpoint:
                        local_l1_alignment_losses[client].load_state_dict(modality_checkpoint['l1_alignment_state_dict'])
                    print(f"Loaded modality model for client {client}")
    
    # Training loop
    weights = np.array(num_train_samples_clients) / np.sum(num_train_samples_clients)
    weights_tensor = torch.tensor(weights, dtype=torch.float32).to(device)

    for round in range(start_round, num_rounds):
        if round % 5 == 0:
            torch.cuda.empty_cache()
        
        print(f"\n{'='*60}\nRound {round+1}/{num_rounds} - Method: {fed_method.upper()}\n{'='*60}")
        
        # Client training
        client_unets = [copy.deepcopy(global_unet) for _ in range(len(client_list))]
        train_results = []
        client_statistics = []
        client_l1_statistics = []
        val_accuracies = []
        for i, client_id in enumerate(client_list):
            print(f"\nTraining Client {client_id}...")
            
            client_unet = client_unets[i].to(device)
            client_modality_model = local_modality_models[client_id].to(device)
            client_alignment = local_alignment_losses[client_id].to(device)
            
            # Optimizers
            unet_optimizer = optim.Adam(client_unet.parameters(), lr=starting_lr, weight_decay=1e-5)
            modality_optimizer = local_modality_optimizers[client_id]

            # Validate before training
            val_acc = helper_validate(
                client_unet,
                client_modality_model,
                val_dataloaders[client_id],
                device,
                criterion,
                use_amp=use_amp,
                amp_dtype=amp_dtype
            )
            print(f"Client {client_id} - Pre-Training Val Acc: {val_acc:.4f}")
            
            if val_acc > best_local_acc[client_id]:
                best_local_acc[client_id] = val_acc
                print(f"New best personalized model for client {client_id} before training! Acc: {val_acc:.4f}")
                
                torch.save({
                    'round': round,
                    'model_state_dict': client_unet.state_dict(),
                    'val_acc': val_acc
                }, model_save_path + f"_personalized_unet_client_{client_id}.pth")
                
                torch.save({
                    'round': round,
                    'model_state_dict': client_modality_model.state_dict(),
                    'alignment_state_dict': client_alignment.state_dict(),
                    'val_acc': val_acc
                }, model_save_path + f"_personalized_modality_client_{client_id}.pth")
            
            # Train with method-specific parameters
            result = helper_train_with_alignment(
                client_unet,
                client_modality_model,
                client_alignment,
                train_dataloaders[client_id],
                unet_optimizer,
                modality_optimizer,
                criterion,
                device,
                alignment_weight=align_weights[str(client_id)],
                collect_stats=True,
                l1_alignment_loss=local_l1_alignment_losses[client_id],
                l1_alignment_weight=align_weights[str(client_id)],
                fed_method=fed_method,
                global_unet=global_unet,
                prev_unet=prev_local_unets[client_id],
                mu=mu,
                temperature=temperature,
                use_amp=use_amp,
                amp_dtype=amp_dtype
            )
            
            train_results.append(result)
            client_statistics.append(result['statistics'])
            client_l1_statistics.append(result['l1_statistics'])
            
            log_str = f"Client {client_id} - Task Loss: {result['task_loss']:.4f}, Align Loss: {result['alignment_loss']:.4f}"
            if fed_method == 'fedprox':
                log_str += f", Prox Loss: {result['prox_loss']:.4f}"
            if fed_method == 'moon':
                log_str += f", MOON Loss: {result['moon_loss']:.4f}"
            print(log_str)
            
            # Validate
            val_acc = helper_validate(
                client_unet,
                client_modality_model,
                val_dataloaders[client_id],
                device,
                criterion,
                use_amp=use_amp,
                amp_dtype=amp_dtype
            )
            val_accuracies.append(val_acc)
            print(f"Client {client_id} - Val Acc: {val_acc:.4f}")
            
            # Save personalized models if best
            if val_acc > best_local_acc[client_id]:
                best_local_acc[client_id] = val_acc
                print(f"New best personalized model for client {client_id}! Acc: {val_acc:.4f}")
                
                torch.save({
                    'round': round,
                    'model_state_dict': client_unet.state_dict(),
                    'val_acc': val_acc
                }, model_save_path + f"_personalized_unet_client_{client_id}.pth")
                
                torch.save({
                    'round': round,
                    'model_state_dict': client_modality_model.state_dict(),
                    'alignment_state_dict': client_alignment.state_dict(),
                    'val_acc': val_acc
                }, model_save_path + f"_personalized_modality_client_{client_id}.pth")
            
            # Store previous local model for MOON
            if fed_method == 'moon':
                prev_local_unets[client_id] = copy.deepcopy(client_unet).to(device)
        
        # Calculate average validation accuracy
        avg_val_acc = np.mean(val_accuracies)
        print(f"\nAverage Validation Accuracy: {avg_val_acc:.4f}")
        
        # Early stopping
        if avg_val_acc > best_avg_val_acc:
            best_avg_val_acc = avg_val_acc
            patience_counter = 0
        else:
            patience_counter += 1
            print(f"No improvement. Patience: {patience_counter}/{patience}")
            
            if patience_counter >= patience:
                print(f"\n{'='*60}")
                print(f"Early stopping triggered!")
                print(f"Best average validation accuracy: {best_avg_val_acc:.4f}")
                print(f"{'='*60}")
                break
        
        # Server aggregation (FedAvg for all methods)
        print("\nAggregating models...")
        global_unet_prev = copy.deepcopy(global_unet)
        global_unet = aggregate_models(
            client_unets, 
            weights_tensor, 
            device=device,
            global_model_prev=global_unet_prev,
            current_round=round,
            cosine_agg_method=cosine_agg_method,
            cosine_start_round=cosine_start_round
        )

        # Aggregate statistics
        global_stats = aggregate_statistics(client_statistics, weights)
        global_l1_stats = aggregate_statistics(client_l1_statistics, weights)
        
        if global_stats is not None:
            for client in client_list:
                if hasattr(local_alignment_losses[client], 'update_global_statistics'):
                    local_alignment_losses[client].update_global_statistics(global_stats)
        
        if global_l1_stats is not None:
            for client in client_list:
                if hasattr(local_l1_alignment_losses[client], 'update_global_statistics'):
                    local_l1_alignment_losses[client].update_global_statistics(global_l1_stats)
        
        # Evaluate the global model on all clients
        print("\nEvaluating Global Model on all clients...")
        val_acc_global = 0.0
        for i, client_id in enumerate(client_list):
            client_modality_model = local_modality_models[client_id].to(device)
            val_acc = helper_validate(
                global_unet,
                client_modality_model,
                val_dataloaders[client_id],
                device,
                criterion,
                use_amp=use_amp,
                amp_dtype=amp_dtype
            )
            print(f"Client {client_id} - Global Model Val Acc: {val_acc:.4f}")
            val_acc_global = val_acc_global + val_acc
        val_acc_global = val_acc_global / len(client_list)
        print(f"Average Global Model Validation Accuracy: {val_acc_global:.4f}")
        if val_acc_global > best_acc:
            best_acc = val_acc_global
            print(f"New best global model! Acc: {best_acc:.4f}")
            torch.save({
                'round': round,
                'model_state_dict': global_unet.state_dict(),
                'val_acc': best_acc,
            }, model_save_path + f"_best_unet_client_global.pth")
            # Also save modality models
            for client_id in client_list:
                torch.save({
                    'round': round,
                    'model_state_dict': local_modality_models[client_id].state_dict(),
                    'alignment_state_dict': local_alignment_losses[client_id].state_dict()
                }, model_save_path + f"_best_modality_client_{client_id}_global.pth")

        # Save checkpoint every 5 rounds
        if (round + 1) % 5 == 0:
            torch.save({
                'round': round,
                'model_state_dict': global_unet.state_dict(),
                'val_acc': best_acc
            }, model_save_path + f"_unet_round_{round}_client_global.pth")
            
            for client_id in client_list:
                torch.save({
                    'round': round,
                    'model_state_dict': local_modality_models[client_id].state_dict(),
                    'alignment_state_dict': local_alignment_losses[client_id].state_dict(),
                    'l1_alignment_state_dict': local_l1_alignment_losses[client_id].state_dict()
                }, model_save_path + f"_modality_round_{round}_client_{client_id}.pth")
            
            # Remove previous checkpoint to save space
            if round >= 5:
                prev_round = round - 5
                prev_unet_path = model_save_path + f"_unet_round_{prev_round}_client_global.pth"
                if os.path.exists(prev_unet_path):
                    os.remove(prev_unet_path)
                for client_id in client_list:
                    prev_modality_path = model_save_path + f"_modality_round_{prev_round}_client_{client_id}.pth"
                    if os.path.exists(prev_modality_path):
                        os.remove(prev_modality_path)
    
    print("\nTraining completed!")
    print(f"Best average validation accuracy: {best_avg_val_acc:.4f}")
    print(f"Best personalized accuracies:")
    for client_id in client_list:
        print(f"  Client {client_id}: {best_local_acc[client_id]:.4f}")

    print("\n" + "="*60)
    print("Training completed!")
    print("="*60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--clients', type=str, default="4,6,13,20")
    parser.add_argument('--num_rounds', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--img_size', type=int, default=224)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--csv_path', type=str, required=True)
    parser.add_argument('--starting_lr', type=float, default=1e-4)
    parser.add_argument('--save_dir', type=str, default='./checkpoints/cnet_stat_proto/')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--checkpoint', type=int, default=None)
    parser.add_argument('--alignment_type', type=str, default='statistics',
                       choices=['statistics', 'prototypes', 'combined', 'higher_order', 
                               'histogram', 'spectral', 'all_stats'])
    parser.add_argument('--alignment_weight', type=float, default=0.1)
    parser.add_argument('--l1_alignment_weight', type=float, default=0.1)
    parser.add_argument('--cosine_agg_method', type=str, default=None,
                       choices=[None, 'layer_wise', 'filter_wise_client', 'filter_wise_global'])
    parser.add_argument('--cosine_start_round', type=int, default=11)
    
    # FedProx and MOON
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
    
    args = parser.parse_args()

    setup_tee_logging(args.log_dir, args.log_file)
    
    client_list = [int(c) for c in args.clients.split(',')]
    device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    
    print(f"Using device: {device}")
    print(f"Federated method: {args.fed_method.upper()}")
    if args.fed_method == 'fedprox':
        print(f"FedProx mu: {args.mu}")
    if args.fed_method == 'moon':
        print(f"MOON temperature: {args.temperature}")
    
    amp_dtype = torch.float16 if args.amp_dtype == 'float16' else torch.bfloat16

    federated_train_with_statistical_prototypes(
        client_list=client_list,
        num_rounds=args.num_rounds,
        batch_size=args.batch_size,
        img_size=args.img_size,
        num_workers=args.num_workers,
        csv_path=args.csv_path,
        starting_lr=args.starting_lr,
        save_dir=args.save_dir,
        device=device,
        checkpoint=args.checkpoint,
        alignment_type=args.alignment_type,
        alignment_weight=args.alignment_weight,
        cosine_agg_method=args.cosine_agg_method,
        cosine_start_round=args.cosine_start_round,
        l1_alignment_weight=args.l1_alignment_weight,
        fed_method=args.fed_method,
        mu=args.mu,
        temperature=args.temperature,
        use_amp=args.amp,
        amp_dtype=amp_dtype
    )
