import numpy as np
from medpy.metric import dc, hd95

def compute_binary_dice(gt, pred):

    num_gt = np.sum(gt)
    num_pred = np.sum(pred)

    if num_gt == 0:
        if num_pred == 0:
            return 1
        else:
            return 0
    else:
        return dc(pred, gt)

def compute_binary_mIOU(gt, pred):

    intersection = np.logical_and(gt, pred)
    union = np.logical_or(gt, pred)
    iou_score = (np.sum(intersection) + 1e-5) / (np.sum(union) + 1e-5)

    return iou_score

def compute_binary_HD95(gt, pred):

    num_gt = np.sum(gt)
    num_pred = np.sum(pred)

    if num_gt == 0 and num_pred == 0:
        return 0
    if num_gt == 0 or num_pred == 0:
        return 373.12866
    
    return hd95(pred, gt, (1, 1))

def compute_multi_dice(gt, pred):

    num_classes = gt.shape[0]
    dice = []

    for i in range(num_classes):
        dice.append(compute_binary_dice(gt[i], pred[i]))

    return dice

def compute_multi_mIOU(gt, pred):

    num_classes = gt.shape[0]
    iou = []

    for i in range(num_classes):
        iou.append(compute_binary_mIOU(gt[i], pred[i]))

    return iou

def compute_multi_HD95(gt, pred):

    num_classes = gt.shape[0]
    hd95_list = []

    for i in range(num_classes):
        hd95_list.append(compute_binary_HD95(gt[i], pred[i]))

    return hd95_list

def compute_seg_metrics(gt, pred):

    result = {}
    gt = gt.astype(np.uint8)
    pred = pred.astype(np.uint8)
    
    result['Dice'] = compute_binary_dice(gt, pred)
    result['IoU'] = compute_binary_mIOU(gt, pred)
    result['HD95'] = compute_binary_HD95(gt, pred)

    return result