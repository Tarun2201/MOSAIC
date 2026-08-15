import torch
import numpy as np
import torch.nn.functional as F

import itertools
    
class Multiclass_SupConLoss(torch.nn.Module):
    """
    Multiclass Supervised Contrastive Loss.

    This loss extends supervised contrastive learning to the multi-label/multi-class setting.
    It supports both cosine similarity and dot product as similarity metrics, and computes
    contrastive loss.

    References:
        - Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf
        - Multilabel Contrastive Learning (equation 6): https://ojs.aaai.org/index.php/AAAI/article/view/29619

    Args:
        temperature (float): Scaling factor for the logits (default: 0.07)
        contrast_mode (str): Either 'one' (only first view as anchor) or 'all' (all views as anchor)
        base_temperature (float): Base temperature for scaling (used in final loss scaling)
        use_cosine_similarity (bool): Whether to use cosine similarity instead of dot product
    """

    def __init__(self, temperature=0.07, contrast_mode='all',
                base_temperature=0.07, use_cosine_similarity=False):
        super(Multiclass_SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature
        self.use_cosine_similarity = use_cosine_similarity

        
    def forward(self, zis, zjs, labels=None, mask=None):
        """
        Forward pass for computing the supervised contrastive loss.

        Args:
            zis (Tensor): features of shape [batch_size, feature_dim]
            zjs (Tensor): features (another view) of shape [batch_size, feature_dim]
            labels (Tensor, optional): Multi-label tensor of shape [batch_size, num_classes].
                                       If provided, a mask is created from labels.
            mask (Tensor, optional): Binary mask of shape [batch_size, batch_size, num_classes],
                                     where mask[i, j, k] = 1 indicates i and j share class k.
                                     Provide either `labels` or `mask`, not both.

        Returns:
            torch.Tensor: A scalar tensor representing the contrastive loss.
        """
        device = torch.device('cuda')
        
        features = torch.cat([zis.unsqueeze(1), zjs.unsqueeze(1)], dim=1)

        if len(features.shape) < 3:
            raise ValueError("`features` needs to be [batch_size, n_views, feature_dim]")
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            # Default to identity mask (only self-positives)
            mask = torch.eye(batch_size, dtype=torch.float32)
            mask = mask.unsqueeze(2).to(device)
            label_dim = 1
        elif labels is not None:
            if len(labels.shape) == 1:
                labels = labels.unsqueeze(1)
            
            label_dim = labels.shape[1]
            labels = labels.contiguous().view(-1, 1, label_dim)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.permute(1,0,2)).float().to(device)
        else:
            mask = mask.float().to(device)
            label_dim = mask.shape[2]

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError(f"Unknown contrast_mode: {self.contrast_mode}")
        
        logits = anchor_feature, contrast_feature

        if self.use_cosine_similarity:
            cosine_similarity = torch.nn.CosineSimilarity(dim=-1)
            logits = cosine_similarity(anchor_feature.unsqueeze(1), contrast_feature.unsqueeze(0))
            logits = torch.div(logits, self.temperature)
        else:
            anchor_dot_contrast = torch.div(
                torch.matmul(anchor_feature, contrast_feature.T),
                self.temperature)
            logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
            logits = (anchor_dot_contrast - logits_max.detach()).unsqueeze(2)

        # mask to ignore self-contrast cases
        mask = mask.repeat(anchor_count, contrast_count, 1)
        logits_mask = torch.ones_like(mask) - torch.eye(batch_size * anchor_count).unsqueeze(2).to(device)
        mask = mask * logits_mask

        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        p_ij = mask.sum(1)
        mean_log_prob_pos_label= (mask * log_prob).sum(1) / p_ij
        mean_log_prob_pos=mean_log_prob_pos_label.sum(1)/label_dim
        
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()
        
        return loss

class MultiLabelFocalLoss(torch.nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        """
        Multi-label Focal Loss for multi-hot encoded targets.

        Args:
            alpha (float or list): Weighting factor for classes. Can be a scalar or a list/array of weights for each class.
            gamma (float): Focusing parameter to adjust the weight given to hard examples.
            reduction (str): Specifies the reduction to apply to the output ('none', 'mean', 'sum').
        """
        super(MultiLabelFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        Args:
            inputs: Logits (unnormalized scores) of shape (batch_size, num_classes).
            targets: Multi-hot encoded labels of shape (batch_size, num_classes).
        Returns:
            Loss value (scalar or tensor depending on the reduction method).
        """
        # Apply sigmoid to get probabilities for multi-label classification
        probs = torch.sigmoid(inputs)
        
        # Ensure targets are on the same device as inputs
        targets = targets.to(inputs.device)
        
        # Compute focal weights
        focal_weight = torch.where(targets == 1, 1 - probs, probs) ** self.gamma

        # Compute log probabilities
        log_probs = torch.where(targets == 1, torch.log(probs + 1e-8), torch.log(1 - probs + 1e-8))

        # Apply alpha weighting
        if isinstance(self.alpha, (float, int)):
            alpha_weight = self.alpha
        elif isinstance(self.alpha, (list, torch.Tensor)):
            alpha_weight = torch.tensor(self.alpha, device=inputs.device)
            alpha_weight = alpha_weight.unsqueeze(0)  # Broadcast for batch
        else:
            raise ValueError("alpha should be a float, int, list, or tensor")

        loss = -alpha_weight * focal_weight * log_probs

        # Apply reduction
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss

class AggrementLoss(torch.nn.Module):
    """Agreement Loss between binary and multiclass CAMs.
    This loss encourages consistency between a binary class activation map (CAM)
    and the most activated class in a multiclass CAM tensor.
    """
    def __init__(self, per_class=False):
        super(AggrementLoss, self).__init__()
        # When True, apply agreement loss to each class independently
        # to ensure all classes (core and edema) get gradient supervision
        self.per_class = per_class

    def forward(self, binary_cam, class_cams):

        """
        Forward pass for computing agreement loss.

        Args:
            binary_cam (Tensor): Binary CAM tensor of shape [B, 1, H, W].
            class_cams (Tensor): Multiclass CAM tensor of shape [B, num_classes, H, W],
                                 where C is the number of classes.

        Returns:
            torch.Tensor: Scalar loss representing binary cross-entropy
                          between the binary CAM and the max-activated
                          class CAM at each pixel.
        """
        
        # --- Detach binary CAM to prevent gradient flow ---
        binary_cam = binary_cam.detach()

        if self.per_class:
            # Per-class agreement: supervise each class independently
            loss = 0
            num_classes = class_cams.shape[1]
            for c in range(num_classes):
                class_cam = class_cams[:, c:c+1, :, :]
                loss += F.binary_cross_entropy_with_logits(class_cam, binary_cam)
            loss = loss / num_classes
        else:
            # Original: only the max-activated class gets supervision
            # --- Select the Class with the Highest Probability at Each Pixel ---
            max_class_cam, _ = torch.max(class_cams, dim=1, keepdim=True)  # [batch, 1, H, W]

            # --- Compute BCE Loss ---
            loss = F.binary_cross_entropy_with_logits(max_class_cam, binary_cam)

        return loss

class SimMinLoss(torch.nn.Module):

    """
    Minimizes cosine similarity between foreground and background embeddings of a class or foreground embeddings of two different classes.
    Used to push within class or between class representations apart in a multi-class setting.
    Sim-minimization term in Class-Specific Separation Loss or Inter-Class Separability Loss.
    """

    def __init__(self, metric='cos', reduction='mean',intra=True):

        """
        Args:
            metric (str): Similarity metric to use ('cos' only supported).
            reduction (str): 'mean' or 'sum' for aggregating loss.
            intra (bool): Whether to use intra-class (foreground vs background) or inter-class (foreground vs foreground) similarity.
        """

        super(SimMinLoss, self).__init__()
        self.metric = metric
        self.reduction = reduction
        self.intra=intra
        
    def forward(self, embedded_fg, embedded_bg):

        """
        Args:
            embedded_fg (Tensor): Foreground embeddings [N, num_classes, feature_dim].
            embedded_bg (Tensor): Background embeddings [N, num_classes, feature_dim].

        Returns:
            torch.Tensor: Similarity minimization loss.
        """

        if self.metric == 'cos':
            sim = cos_simi(embedded_fg, embedded_bg, self.intra)
            loss = -torch.log(1 - sim)
        
        else:
            raise NotImplementedError

        if self.reduction == 'mean':    
            return torch.mean(loss)
        elif self.reduction == 'sum':
            return torch.sum(loss)
        else:
            raise ValueError("Reduction must be 'mean' or 'sum'.")

class SimMaxLoss_intraclass(torch.nn.Module):

    """
    Maximizes cosine similarity between foregroud-foreground or background-background embeddings of a class
    Used to pull same-class features closer.
    Sim-maximization term in Class-Specific Separation Loss.
    """

    def __init__(self, metric='cos', alpha=0.25, reduction='mean'):

        """
        Args:
            metric (str): Similarity metric to use ('cos' only supported).
            alpha (float): Rank-based weighting factor.
            reduction (str): 'mean' or 'sum' for aggregating loss.
        """

        super(SimMaxLoss_intraclass, self).__init__()
        self.metric = metric
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, embedded_bg):
        
        """
        Args:
            embedded_bg (Tensor): Foreground or Background embeddings [N, num_classes, feature_dim].

        Returns:
            torch.Tensor: Similarity maximization loss.
        """

        if self.metric == 'cos':
            sim = cos_simi(embedded_bg, embedded_bg)
            loss = -torch.log(sim)
            loss[loss < 0] = 0
            
            _, indices = sim.sort(descending=True, dim=2)
            _, rank = indices.sort(dim=2)
            rank = rank - 1
            
            rank_weights = torch.exp(-rank.float() * self.alpha)
            loss = loss * rank_weights

        else:
            raise NotImplementedError

        if self.reduction == 'mean':
            return torch.mean(loss)
            
        elif self.reduction == 'sum':
            return torch.sum(loss)

def cos_simi(embedded_fg, embedded_bg, intra=True):

    """
    Computes cosine similarities between two sets of embeddings.

    Args:
        embedded_fg (Tensor): Foreground embeddings [N, num_classes, feature_dim].
        embedded_bg (Tensor): Background embeddings [N, num_classes, feature_dim].
        intra (bool): Whether to use intra-class (foreground vs background) or inter-class (foreground vs foreground) similarity.

    Returns:
        Tensor: Cosine similarity values.
    """

    embedded_fg = F.normalize(embedded_fg, dim=2)
    embedded_bg = F.normalize(embedded_bg, dim=2)
    
    if intra:
        embedded_fg = embedded_fg.permute(1, 0, 2)
        embedded_bg = embedded_bg.permute(1, 0, 2)
        sim = torch.bmm(embedded_fg, embedded_bg.permute(0, 2, 1))
        sim = sim.permute(1, 0, 2)

    else:
        pairs = list(itertools.combinations(range(embedded_fg.size(1)), 2))
        sim = [torch.bmm(embedded_fg[:, pair[0]].unsqueeze(1), embedded_bg[:, pair[1]].unsqueeze(2)) for pair in pairs]
        sim = torch.stack(sim, dim=1)

    return torch.clamp(sim, min=0.0005, max=0.9995)





