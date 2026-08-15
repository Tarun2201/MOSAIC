import numpy as np
from scipy.ndimage import distance_transform_edt
from scipy.spatial.distance import directed_hausdorff


def compute_3d_dice(gt_volume, pred_volume):
    """
    Compute 3D Dice coefficient for a volume
    
    Args:
        gt_volume: numpy array of shape (D, H, W) - ground truth
        pred_volume: numpy array of shape (D, H, W) - prediction
    
    Returns:
        dice_score: float
    """
    gt_volume = gt_volume.astype(np.bool_)
    pred_volume = pred_volume.astype(np.bool_)
    
    intersection = np.logical_and(gt_volume, pred_volume).sum()
    gt_sum = gt_volume.sum()
    pred_sum = pred_volume.sum()
    
    if gt_sum + pred_sum == 0:
        return 1.0  # Both empty, perfect match
    
    dice = (2.0 * intersection) / (gt_sum + pred_sum)
    return dice


def compute_3d_iou(gt_volume, pred_volume):
    """
    Compute 3D IoU (Jaccard index) for a volume
    
    Args:
        gt_volume: numpy array of shape (D, H, W)
        pred_volume: numpy array of shape (D, H, W)
    
    Returns:
        iou_score: float
    """
    gt_volume = gt_volume.astype(np.bool_)
    pred_volume = pred_volume.astype(np.bool_)
    
    intersection = np.logical_and(gt_volume, pred_volume).sum()
    union = np.logical_or(gt_volume, pred_volume).sum()
    
    if union == 0:
        return 1.0  # Both empty
    
    iou = intersection / union
    return iou


def compute_3d_hd95(gt_volume, pred_volume, voxel_spacing=(1.0, 1.0, 1.0)):
    """
    Compute 3D Hausdorff Distance 95th percentile for a volume
    
    Args:
        gt_volume: numpy array of shape (D, H, W)
        pred_volume: numpy array of shape (D, H, W)
        voxel_spacing: tuple of (z_spacing, y_spacing, x_spacing)
    
    Returns:
        hd95: float (in mm if voxel_spacing is in mm)
    """
    gt_volume = gt_volume.astype(np.bool_)
    pred_volume = pred_volume.astype(np.bool_)
    
    # Check if either is empty
    if gt_volume.sum() == 0 or pred_volume.sum() == 0:
        if gt_volume.sum() == pred_volume.sum():
            return 0.0  # Both empty
        else:
            # One empty, one not - return large distance
            return 373.12866  # Default large value (diagonal of typical volume)
    
    # Get surface voxels
    gt_surface = get_surface_voxels(gt_volume)
    pred_surface = get_surface_voxels(pred_volume)
    
    if gt_surface.sum() == 0 or pred_surface.sum() == 0:
        return 373.12866
    
    # Get coordinates of surface voxels
    gt_coords = np.array(np.where(gt_surface)).T * np.array(voxel_spacing)
    pred_coords = np.array(np.where(pred_surface)).T * np.array(voxel_spacing)
    
    # Compute distances from gt to pred
    distances_gt_to_pred = []
    for gt_point in gt_coords:
        dists = np.sqrt(np.sum((pred_coords - gt_point)**2, axis=1))
        distances_gt_to_pred.append(dists.min())
    
    # Compute distances from pred to gt
    distances_pred_to_gt = []
    for pred_point in pred_coords:
        dists = np.sqrt(np.sum((gt_coords - pred_point)**2, axis=1))
        distances_pred_to_gt.append(dists.min())
    
    # Combine and compute 95th percentile
    all_distances = distances_gt_to_pred + distances_pred_to_gt
    hd95 = np.percentile(all_distances, 95)
    
    return hd95


def get_surface_voxels(volume):
    """
    Extract surface voxels (voxels at the boundary)
    
    Args:
        volume: binary numpy array of shape (D, H, W)
    
    Returns:
        surface: binary numpy array of same shape
    """
    # Erode by 1 voxel
    from scipy.ndimage import binary_erosion
    eroded = binary_erosion(volume)
    
    # Surface is original minus eroded
    surface = np.logical_and(volume, np.logical_not(eroded))
    
    return surface


def compute_3d_volume_metrics(gt_volume, pred_volume, voxel_spacing=(1.0, 1.0, 1.0)):
    """
    Compute all 3D metrics for a volume
    
    Args:
        gt_volume: numpy array of shape (D, H, W)
        pred_volume: numpy array of shape (D, H, W)
        voxel_spacing: tuple of (z_spacing, y_spacing, x_spacing)
    
    Returns:
        dict with 'Dice', 'IoU', 'HD95' keys
    """
    result = {}
    
    gt_volume = gt_volume.astype(np.uint8)
    pred_volume = pred_volume.astype(np.uint8)
    
    result['Dice'] = compute_3d_dice(gt_volume, pred_volume)
    result['IoU'] = compute_3d_iou(gt_volume, pred_volume)
    result['HD95'] = compute_3d_hd95(gt_volume, pred_volume, voxel_spacing)
    
    return result


def compute_multiclass_3d_metrics(gt_volume, pred_volume, num_classes, voxel_spacing=(1.0, 1.0, 1.0)):
    """
    Compute 3D metrics for multi-class segmentation
    
    Args:
        gt_volume: numpy array of shape (num_classes, D, H, W)
        pred_volume: numpy array of shape (num_classes, D, H, W)
        num_classes: int
        voxel_spacing: tuple
    
    Returns:
        list of dicts, one per class
    """
    results = []
    
    for i in range(num_classes):
        result = compute_3d_volume_metrics(gt_volume[i], pred_volume[i], voxel_spacing)
        results.append(result)
    
    return results
