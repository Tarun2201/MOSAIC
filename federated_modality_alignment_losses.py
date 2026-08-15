import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class FeatureStatisticsAlignmentLoss(nn.Module):
    """
    Approach 1: Feature Statistics Alignment
    
    Each client computes mean and std of their modality features.
    Server aggregates these statistics across clients.
    Clients then align their features to the global statistics.
    
    Privacy-preserving: Only statistics (mean/std) are shared, not raw data.
    """
    def __init__(self, num_channels=3, momentum=0.9):
        super().__init__()
        self.num_channels = num_channels
        self.momentum = momentum
        
        # Global statistics (updated from server)
        self.register_buffer('global_mean', torch.zeros(num_channels))
        self.register_buffer('global_std', torch.ones(num_channels))
        
    def compute_local_statistics(self, features):
        """
        Compute mean and std for this batch
        features: [B, C, H, W]
        Returns: dict with 'mean' and 'std' of shape [C]
        """
        mean = features.mean(dim=[0, 2, 3])  # [C]
        std = features.std(dim=[0, 2, 3])    # [C]
        return {'mean': mean.detach(), 'std': std.detach()}
    
    def update_global_statistics(self, global_stats):
        """
        Called by server to update global statistics
        Uses momentum for smooth updates
        
        Args:
            global_stats: dict with 'mean' and 'std' keys
        """
        if 'mean' in global_stats:
            self.global_mean = self.momentum * self.global_mean + (1 - self.momentum) * global_stats['mean']
        if 'std' in global_stats:
            self.global_std = self.momentum * self.global_std + (1 - self.momentum) * global_stats['std']
    
    def forward(self, features):
        """
        Compute alignment loss between local features and global statistics
        features: [B, C, H, W]
        """
        local_mean = features.mean(dim=[0, 2, 3])  # [C]
        local_std = features.std(dim=[0, 2, 3])    # [C]
        
        mean_loss = F.mse_loss(local_mean, self.global_mean)
        std_loss = F.mse_loss(local_std, self.global_std)
        
        return mean_loss + std_loss


class PrototypeAlignmentLoss(nn.Module):
    """
    Approach 2: Learnable Prototypes
    
    Server maintains learnable prototypes representing canonical 3-channel features.
    Clients align their modality features to these shared prototypes.
    Prototypes are updated via FedAvg like model parameters.
    
    Privacy-preserving: Only prototype gradients/updates are shared.
    """
    def __init__(self, num_prototypes=16, feature_dim=3, temperature=0.1):
        super().__init__()
        self.num_prototypes = num_prototypes
        self.temperature = temperature
        
        # Shared prototypes (aggregated across clients)
        self.prototypes = nn.Parameter(torch.randn(num_prototypes, feature_dim))
        nn.init.xavier_uniform_(self.prototypes)
        
    def forward(self, features):
        """
        Align features to prototypes
        features: [B, C, H, W]
        """
        B, C, H, W = features.shape
        
        # Reshape features to [B*H*W, C]
        features_flat = features.permute(0, 2, 3, 1).reshape(-1, C)
        
        # Normalize
        prototypes_norm = F.normalize(self.prototypes, dim=1)
        features_norm = F.normalize(features_flat, dim=1)
        
        # Compute similarity
        similarity = torch.matmul(features_norm, prototypes_norm.T) / self.temperature
        
        # Soft assignment (each feature aligns to prototypes)
        assignment = F.softmax(similarity, dim=1)  # [B*H*W, num_prototypes]
        
        # Reconstruct features from prototypes
        reconstructed = torch.matmul(assignment, self.prototypes)
        
        # Reconstruction loss
        reconstruction_loss = F.mse_loss(reconstructed, features_flat)
        
        # Diversity loss (encourage prototypes to be different)
        proto_similarity = torch.matmul(prototypes_norm, prototypes_norm.T)
        diversity_loss = (proto_similarity - torch.eye(self.num_prototypes, device=proto_similarity.device)).pow(2).mean()
        
        return reconstruction_loss + 0.1 * diversity_loss


class ConsistencyRegularizationLoss(nn.Module):
    """
    Approach 3: Consistency Regularization via Pseudo-Labeling
    
    Use the global classifier to generate pseudo-labels on each client.
    Encourage modality features that lead to consistent predictions with global model.
    
    Privacy-preserving: Only model parameters shared, not data.
    """
    def __init__(self, temperature=0.5):
        super().__init__()
        self.temperature = temperature
        
    def forward(self, features_local, features_with_global, predictions_global):
        """
        Encourage local modality features to produce similar predictions as global
        
        features_local: features from local modality model [B, C, H, W]
        features_with_global: features when using aggregated/reference features [B, C, H, W]
        predictions_global: predictions from global classifier
        """
        # L2 distance between feature distributions
        consistency_loss = F.mse_loss(features_local, features_with_global)
        
        return consistency_loss


class DistributionMatchingLoss(nn.Module):
    """
    Approach 4: Maximum Mean Discrepancy (MMD)
    
    Matches distributions of features across clients without sharing raw data.
    Each client computes kernel embeddings, shares only aggregated statistics.
    
    Privacy-preserving: Kernel statistics aggregated, not raw features.
    """
    def __init__(self, kernel_type='gaussian', kernel_mul=2.0, kernel_num=5):
        super().__init__()
        self.kernel_type = kernel_type
        self.kernel_mul = kernel_mul
        self.kernel_num = kernel_num
        
        # Store global feature statistics
        self.register_buffer('global_kernel_mean', None)
        
    def gaussian_kernel(self, x, y, sigma=1.0):
        """
        Compute Gaussian kernel matrix
        """
        x_size = x.size(0)
        y_size = y.size(0)
        dim = x.size(1)
        
        x = x.unsqueeze(1)  # [x_size, 1, dim]
        y = y.unsqueeze(0)  # [1, y_size, dim]
        
        tiled_x = x.expand(x_size, y_size, dim)
        tiled_y = y.expand(x_size, y_size, dim)
        
        kernel_input = (tiled_x - tiled_y).pow(2).mean(2) / sigma
        return torch.exp(-kernel_input)
    
    def compute_kernel_statistics(self, features):
        """
        Compute kernel-based statistics for features
        features: [B, C, H, W]
        Returns: kernel mean that can be aggregated
        """
        B, C, H, W = features.shape
        features_flat = features.reshape(B, -1)
        
        # Compute self-kernel
        K = self.gaussian_kernel(features_flat, features_flat)
        kernel_mean = K.mean()
        
        return kernel_mean
    
    def forward(self, features):
        """
        Compute MMD-based loss
        """
        if self.global_kernel_mean is None:
            # Initialize with current batch
            self.global_kernel_mean = self.compute_kernel_statistics(features)
            return torch.tensor(0.0, device=features.device)
        
        local_kernel_mean = self.compute_kernel_statistics(features)
        
        # Minimize difference from global kernel statistics
        mmd_loss = (local_kernel_mean - self.global_kernel_mean).pow(2)
        
        return mmd_loss


class ModalityInvariantNormalization(nn.Module):
    """
    Approach 5: Learnable Normalization Layer
    
    Adds a trainable normalization that's shared across clients.
    Each client's modality model output is normalized to a common space.
    
    Privacy-preserving: Only normalization parameters shared.
    """
    def __init__(self, num_channels=3, momentum=0.1):
        super().__init__()
        self.norm = nn.InstanceNorm2d(num_channels, affine=True, track_running_stats=True, momentum=momentum)
        
    def forward(self, x):
        """
        Normalize features to common space
        x: [B, C, H, W]
        """
        return self.norm(x)


class ContrastiveCrossClientLoss(nn.Module):
    """
    Approach 6: Server-Side Feature Queue for Contrastive Learning
    
    Server maintains a queue of aggregated feature statistics from all clients.
    Each client pulls from this queue to compute contrastive loss.
    
    Privacy-preserving: Only aggregated feature representations shared.
    """
    def __init__(self, feature_dim=3, queue_size=1024, temperature=0.07):
        super().__init__()
        self.feature_dim = feature_dim
        self.queue_size = queue_size
        self.temperature = temperature
        
        # Server-side feature queue (aggregated representations)
        self.register_buffer('feature_queue', torch.randn(queue_size, feature_dim))
        self.feature_queue = F.normalize(self.feature_queue, dim=1)
        
        self.register_buffer('queue_ptr', torch.zeros(1, dtype=torch.long))
        
    @torch.no_grad()
    def update_queue(self, features):
        """
        Update server-side feature queue with aggregated features
        Called after each round with aggregated client features
        """
        batch_size = features.shape[0]
        
        ptr = int(self.queue_ptr)
        
        # Replace oldest features
        if ptr + batch_size <= self.queue_size:
            self.feature_queue[ptr:ptr + batch_size] = features
        else:
            # Wrap around
            remaining = self.queue_size - ptr
            self.feature_queue[ptr:] = features[:remaining]
            self.feature_queue[:batch_size - remaining] = features[remaining:]
        
        ptr = (ptr + batch_size) % self.queue_size
        self.queue_ptr[0] = ptr
    
    def forward(self, features):
        """
        Contrastive loss: pull features towards queue (global features)
        features: [B, C, H, W]
        """
        B, C, H, W = features.shape
        
        # Aggregate features spatially
        features_pooled = F.adaptive_avg_pool2d(features, (1, 1)).squeeze()  # [B, C]
        features_pooled = F.normalize(features_pooled, dim=1)
        
        # Compute similarity to queue
        similarity = torch.matmul(features_pooled, self.feature_queue.T) / self.temperature  # [B, queue_size]
        
        # Encourage high similarity to all queue items (pull towards global distribution)
        # Use margin-based loss
        target_similarity = 0.8  # Target similarity
        loss = F.relu(target_similarity - similarity).mean()
        
        return loss


class CombinedModalityAlignmentLoss(nn.Module):
    """
    Combined loss that uses multiple alignment strategies
    """
    def __init__(self, use_statistics=True, use_prototypes=True, use_normalization=True,
                 lambda_stats=1.0, lambda_proto=0.5, lambda_norm=0.1):
        super().__init__()
        
        self.use_statistics = use_statistics
        self.use_prototypes = use_prototypes
        self.use_normalization = use_normalization
        
        self.lambda_stats = lambda_stats
        self.lambda_proto = lambda_proto
        self.lambda_norm = lambda_norm
        
        if use_statistics:
            self.stats_loss = FeatureStatisticsAlignmentLoss()
        
        if use_prototypes:
            self.proto_loss = PrototypeAlignmentLoss()
        
        if use_normalization:
            self.norm = ModalityInvariantNormalization()
    
    def forward(self, features):
        """
        Compute combined alignment loss
        """
        total_loss = 0.0
        loss_dict = {}
        
        # Apply normalization first if enabled
        if self.use_normalization:
            features = self.norm(features)
        
        # Statistics alignment
        if self.use_statistics:
            stats_loss = self.stats_loss(features)
            total_loss += self.lambda_stats * stats_loss
            loss_dict['stats_alignment'] = stats_loss.item()
        
        # Prototype alignment
        if self.use_prototypes:
            proto_loss = self.proto_loss(features)
            total_loss += self.lambda_proto * proto_loss
            loss_dict['proto_alignment'] = proto_loss.item()
        
        return total_loss, loss_dict
    
    def get_local_statistics(self, features):
        """
        Get statistics to send to server
        """
        if self.use_statistics:
            return self.stats_loss.compute_local_statistics(features)
        return None
    
    def update_from_server(self, global_stats=None, global_prototypes=None):
        """
        Update local modules with global information from server
        """
        if global_stats is not None and self.use_statistics:
            self.stats_loss.update_global_statistics(global_stats)
        
        if global_prototypes is not None and self.use_prototypes:
            self.proto_loss.prototypes.data = global_prototypes


class HigherOrderStatisticsAlignmentLoss(nn.Module):
    """
    Approach: Higher-Order Statistical Prototypes
    
    Instead of learnable prototypes, use FIXED statistical targets computed from features:
    - Skewness (3rd moment): Measures asymmetry of distribution
    - Kurtosis (4th moment): Measures tail heaviness
    - Percentiles: Distribution shape (25th, 50th, 75th)
    - Energy: L2 norm of features
    - Correlation: Cross-channel correlations
    
    These act as "statistical prototypes" - fixed anchors all clients align to.
    """
    def __init__(self, num_channels=3, momentum=0.9, use_skewness=True, use_kurtosis=True, 
                 use_percentiles=True, use_energy=True, use_correlation=True):
        super().__init__()
        self.num_channels = num_channels
        self.momentum = momentum
        
        self.use_skewness = use_skewness
        self.use_kurtosis = use_kurtosis
        self.use_percentiles = use_percentiles
        self.use_energy = use_energy
        self.use_correlation = use_correlation
        
        # Global statistical prototypes (updated from server)
        if use_skewness:
            self.register_buffer('global_skewness', torch.zeros(num_channels))
        if use_kurtosis:
            self.register_buffer('global_kurtosis', torch.ones(num_channels) * 3)  # Normal dist has kurtosis=3
        if use_percentiles:
            # 25th, 50th (median), 75th percentiles for each channel
            self.register_buffer('global_percentile_25', torch.zeros(num_channels))
            self.register_buffer('global_percentile_50', torch.zeros(num_channels))
            self.register_buffer('global_percentile_75', torch.zeros(num_channels))
        if use_energy:
            self.register_buffer('global_energy', torch.ones(num_channels))
        if use_correlation:
            # Cross-channel correlation matrix (3x3 for RGB)
            self.register_buffer('global_correlation', torch.eye(num_channels))
    
    def compute_skewness(self, x):
        """
        Compute skewness (3rd standardized moment)
        x: [B, C, H, W]
        Returns: [C]
        """
        mean = x.mean(dim=[0, 2, 3], keepdim=True)
        std = x.std(dim=[0, 2, 3], keepdim=True)
        z = (x - mean) / (std + 1e-7)
        skewness = (z ** 3).mean(dim=[0, 2, 3])
        return skewness
    
    def compute_kurtosis(self, x):
        """
        Compute kurtosis (4th standardized moment)
        x: [B, C, H, W]
        Returns: [C]
        """
        mean = x.mean(dim=[0, 2, 3], keepdim=True)
        std = x.std(dim=[0, 2, 3], keepdim=True)
        z = (x - mean) / (std + 1e-7)
        kurtosis = (z ** 4).mean(dim=[0, 2, 3])
        return kurtosis
    
    def compute_percentiles(self, x):
        """
        Compute percentiles per channel
        x: [B, C, H, W]
        Returns: dict with p25, p50, p75 each of shape [C]
        """
        B, C, H, W = x.shape
        x_flat = x.permute(0, 2, 3, 1).reshape(-1, C)  # [B*H*W, C]
        
        p25 = torch.quantile(x_flat, 0.25, dim=0)
        p50 = torch.quantile(x_flat, 0.50, dim=0)
        p75 = torch.quantile(x_flat, 0.75, dim=0)
        
        return {'p25': p25, 'p50': p50, 'p75': p75}
    
    def compute_energy(self, x):
        """
        Compute energy (mean L2 norm) per channel
        x: [B, C, H, W]
        Returns: [C]
        """
        energy = (x ** 2).mean(dim=[0, 2, 3])
        return energy
    
    def compute_correlation(self, x):
        """
        Compute cross-channel correlation matrix
        x: [B, C, H, W]
        Returns: [C, C]
        """
        B, C, H, W = x.shape
        x_flat = x.permute(0, 2, 3, 1).reshape(-1, C)  # [B*H*W, C]
        
        # Normalize
        x_centered = x_flat - x_flat.mean(dim=0, keepdim=True)
        cov = torch.matmul(x_centered.T, x_centered) / x_centered.shape[0]
        
        # Convert to correlation
        std = torch.sqrt(torch.diag(cov))
        correlation = cov / (std.unsqueeze(1) * std.unsqueeze(0) + 1e-7)
        
        return correlation
    
    def compute_local_statistics(self, features):
        """
        Compute all local statistics for this batch
        features: [B, C, H, W]
        Returns: dict with all statistics
        """
        stats = {}
        
        if self.use_skewness:
            stats['skewness'] = self.compute_skewness(features).detach()
        if self.use_kurtosis:
            stats['kurtosis'] = self.compute_kurtosis(features).detach()
        if self.use_percentiles:
            percentiles = self.compute_percentiles(features)
            stats['percentile_25'] = percentiles['p25'].detach()
            stats['percentile_50'] = percentiles['p50'].detach()
            stats['percentile_75'] = percentiles['p75'].detach()
        if self.use_energy:
            stats['energy'] = self.compute_energy(features).detach()
        if self.use_correlation:
            stats['correlation'] = self.compute_correlation(features).detach()
        
        return stats
    
    def update_global_statistics(self, global_stats):
        """
        Update global statistical prototypes from server
        """
        if self.use_skewness and 'skewness' in global_stats:
            self.global_skewness = self.momentum * self.global_skewness + (1 - self.momentum) * global_stats['skewness']
        if self.use_kurtosis and 'kurtosis' in global_stats:
            self.global_kurtosis = self.momentum * self.global_kurtosis + (1 - self.momentum) * global_stats['kurtosis']
        if self.use_percentiles:
            if 'percentile_25' in global_stats:
                self.global_percentile_25 = self.momentum * self.global_percentile_25 + (1 - self.momentum) * global_stats['percentile_25']
            if 'percentile_50' in global_stats:
                self.global_percentile_50 = self.momentum * self.global_percentile_50 + (1 - self.momentum) * global_stats['percentile_50']
            if 'percentile_75' in global_stats:
                self.global_percentile_75 = self.momentum * self.global_percentile_75 + (1 - self.momentum) * global_stats['percentile_75']
        if self.use_energy and 'energy' in global_stats:
            self.global_energy = self.momentum * self.global_energy + (1 - self.momentum) * global_stats['energy']
        if self.use_correlation and 'correlation' in global_stats:
            self.global_correlation = self.momentum * self.global_correlation + (1 - self.momentum) * global_stats['correlation']
    
    def forward(self, features):
        """
        Compute alignment loss between local features and global statistical prototypes
        features: [B, C, H, W]
        """
        total_loss = 0.0
        
        if self.use_skewness:
            local_skewness = self.compute_skewness(features)
            total_loss += F.mse_loss(local_skewness, self.global_skewness)
        
        if self.use_kurtosis:
            local_kurtosis = self.compute_kurtosis(features)
            total_loss += F.mse_loss(local_kurtosis, self.global_kurtosis)
        
        if self.use_percentiles:
            percentiles = self.compute_percentiles(features)
            total_loss += F.mse_loss(percentiles['p25'], self.global_percentile_25)
            total_loss += F.mse_loss(percentiles['p50'], self.global_percentile_50)
            total_loss += F.mse_loss(percentiles['p75'], self.global_percentile_75)
        
        if self.use_energy:
            local_energy = self.compute_energy(features)
            total_loss += F.mse_loss(local_energy, self.global_energy)
        
        if self.use_correlation:
            local_correlation = self.compute_correlation(features)
            total_loss += F.mse_loss(local_correlation, self.global_correlation)
        
        return total_loss


class HistogramPrototypeAlignmentLoss(nn.Module):
    """
    Approach: Histogram-based Statistical Prototypes
    
    Match the distribution of features using histogram bins as prototypes.
    Each bin count acts as a prototype - fixed statistical target.
    
    FEDERATED PRIVACY: Only aggregated histogram statistics are shared with server,
    not raw data or individual samples.
    """
    def __init__(self, num_channels=3, num_bins=32, momentum=0.9, value_range=(-3, 3), sigma=0.01):
        super().__init__()
        self.num_channels = num_channels
        self.num_bins = num_bins
        self.momentum = momentum
        self.value_range = value_range
        self.sigma = sigma  # Smoothing parameter for soft histogram
        
        # Global histogram prototypes for each channel [C, num_bins]
        # These are the ONLY thing shared across clients (aggregated statistics)
        self.register_buffer('global_histogram', torch.ones(num_channels, num_bins) / num_bins)
        
        # Create bin centers
        bin_edges = torch.linspace(value_range[0], value_range[1], num_bins + 1)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        self.register_buffer('bin_centers', bin_centers)
    
    def compute_soft_histogram(self, features):
        """
        Compute differentiable soft histogram for each channel using RBF kernels
        features: [B, C, H, W]
        Returns: [C, num_bins]
        """
        B, C, H, W = features.shape
        histograms = []
        
        for c in range(C):
            channel_data = features[:, c, :, :].flatten()  # [B*H*W]
            
            # Compute distances to bin centers: [B*H*W, num_bins]
            distances = channel_data.unsqueeze(1) - self.bin_centers.unsqueeze(0)
            
            # Apply RBF kernel: exp(-d^2 / (2*sigma^2))
            weights = torch.exp(-distances ** 2 / (2 * self.sigma ** 2))
            
            # Sum weights for each bin and normalize
            hist = weights.sum(dim=0)
            hist = hist / (hist.sum() + 1e-7)  # Normalize to probability distribution
            histograms.append(hist)
        
        return torch.stack(histograms)  # [C, num_bins]
    
    def compute_histogram(self, features):
        """
        Compute normalized histogram for statistics collection (non-differentiable)
        features: [B, C, H, W]
        Returns: [C, num_bins]
        """
        B, C, H, W = features.shape
        histograms = []
        
        for c in range(C):
            channel_data = features[:, c, :, :].flatten()
            hist = torch.histc(channel_data, bins=self.num_bins, 
                              min=self.value_range[0], max=self.value_range[1])
            hist = hist / (hist.sum() + 1e-7)  # Normalize
            histograms.append(hist)
        
        return torch.stack(histograms)  # [C, num_bins]
    
    def compute_local_statistics(self, features):
        """
        Compute histogram statistics
        """
        return {'histogram': self.compute_histogram(features).detach()}
    
    def update_global_statistics(self, global_stats):
        """
        Update global histogram prototypes
        """
        if 'histogram' in global_stats:
            self.global_histogram = self.momentum * self.global_histogram + (1 - self.momentum) * global_stats['histogram']
    
    def forward(self, features):
        """
        Compute histogram matching loss using differentiable soft histogram
        """
        # Use soft histogram for gradient computation
        local_histogram = self.compute_soft_histogram(features)
        
        # KL divergence between histograms: sum(p * log(p/q))
        kl_div = F.kl_div(
            (local_histogram + 1e-7).log(), 
            self.global_histogram + 1e-7, 
            reduction='batchmean'
        )
        
        return kl_div


class SpectralPrototypeAlignmentLoss(nn.Module):
    """
    Approach: Spectral (Frequency Domain) Statistical Prototypes
    
    Use FFT to compute frequency domain statistics as prototypes:
    - Dominant frequencies
    - Spectral energy distribution
    - Frequency band energies
    
    Captures texture and pattern information, complementary to spatial statistics.
    
    FEDERATED PRIVACY: Only aggregated frequency band energies are shared with server,
    not raw data or individual samples.
    """
    def __init__(self, num_channels=3, momentum=0.9, num_freq_bands=8):
        super().__init__()
        self.num_channels = num_channels
        self.momentum = momentum
        self.num_freq_bands = num_freq_bands
        self._eps = 1e-7
        self._band_cache = {}
        
        # Global spectral prototypes [C, num_freq_bands]
        # These are the ONLY thing shared across clients (aggregated statistics)
        self.register_buffer('global_freq_energy', torch.ones(num_channels, num_freq_bands))

    def _get_band_masks(self, height, width, device, dtype):
        cache_key = (height, width, str(device), str(dtype))
        cached = self._band_cache.get(cache_key)
        if cached is not None:
            return cached

        center_y = height // 2
        center_x = width // 2

        y = torch.arange(height, device=device, dtype=torch.float32)
        x = torch.arange(width, device=device, dtype=torch.float32)
        yy, xx = torch.meshgrid(y, x, indexing='ij')
        radius = torch.sqrt((xx - center_x) ** 2 + (yy - center_y) ** 2)

        max_radius = min(height, width) // 2
        band_edges = torch.linspace(0, max_radius, self.num_freq_bands + 1, device=device, dtype=torch.float32)
        r_min = band_edges[:-1]
        r_max = band_edges[1:]

        band_masks = (radius[None, :, :] >= r_min[:, None, None]) & (radius[None, :, :] < r_max[:, None, None])
        band_masks = band_masks.to(dtype)
        band_norms = band_masks.sum(dim=(-2, -1)).clamp_min(1.0).to(dtype)

        self._band_cache[cache_key] = (band_masks, band_norms)
        return band_masks, band_norms
    
    def compute_spectral_energy(self, features):
        """
        Compute frequency band energies using FFT
        features: [B, C, H, W]
        Returns: [C, num_freq_bands]
        """
        _, _, height, width = features.shape

        fft_input = features
        if fft_input.dtype in (torch.float16, torch.bfloat16):
            fft_input = fft_input.float()

        fft = torch.fft.fft2(fft_input, dim=(-2, -1))
        magnitude = torch.abs(fft).mean(dim=0)  # [C, H, W]

        band_masks, band_norms = self._get_band_masks(height, width, features.device, magnitude.dtype)

        band_sums = (magnitude.unsqueeze(0) * band_masks[:, None, :, :]).sum(dim=(-2, -1))
        band_means = band_sums / band_norms[:, None]

        return band_means.transpose(0, 1)  # [C, num_freq_bands]
    
    def compute_local_statistics(self, features):
        """
        Compute spectral statistics
        """
        return {'freq_energy': self.compute_spectral_energy(features).detach()}
    
    def update_global_statistics(self, global_stats):
        """
        Update global spectral prototypes
        """
        if 'freq_energy' in global_stats:
            self.global_freq_energy = self.momentum * self.global_freq_energy + (1 - self.momentum) * global_stats['freq_energy']
    
    def forward(self, features):
        """
        Compute spectral alignment loss
        """
        local_freq_energy = self.compute_spectral_energy(features)
        return F.mse_loss(local_freq_energy, self.global_freq_energy)
