import numpy as np
import torch
from typing import Dict


def dice_coefficient(pred: np.ndarray, target: np.ndarray, smooth: float = 1e-6) -> float:
    """计算 Dice 相似系数 (DSC)"""
    intersection = np.sum(pred * target)
    return (2.0 * intersection + smooth) / (np.sum(pred) + np.sum(target) + smooth)


def iou_score(pred: np.ndarray, target: np.ndarray, smooth: float = 1e-6) -> float:
    """计算交并比 (IOU)"""
    intersection = np.sum(pred * target)
    union = np.sum(pred) + np.sum(target) - intersection
    return (intersection + smooth) / (union + smooth)


def precision_score(pred: np.ndarray, target: np.ndarray, smooth: float = 1e-6) -> float:
    """计算精确率 (Precision)"""
    tp = np.sum(pred * target)
    fp = np.sum(pred * (1 - target))
    return (tp + smooth) / (tp + fp + smooth)


def recall_score(pred: np.ndarray, target: np.ndarray, smooth: float = 1e-6) -> float:
    """计算召回率 (Recall)"""
    tp = np.sum(pred * target)
    fn = np.sum((1 - pred) * target)
    return (tp + smooth) / (tp + fn + smooth)


def specificity_score(pred: np.ndarray, target: np.ndarray, smooth: float = 1e-6) -> float:
    """计算特异性 (Specificity)"""
    tn = np.sum((1 - pred) * (1 - target))
    fp = np.sum(pred * (1 - target))
    return (tn + smooth) / (tn + fp + smooth)


def hausdorff_distance_95(pred: np.ndarray, target: np.ndarray) -> float:
    """计算 95% 豪斯多夫距离 (HD95)
    
    需要安装 medpy: pip install medpy
    """
    if np.sum(pred) == 0 and np.sum(target) == 0:
        return 0.0
    if np.sum(pred) == 0 or np.sum(target) == 0:
        # 使用图像对角线长度作为最大距离
        h, w = pred.shape[-2:]
        return np.sqrt(h**2 + w**2)
    
    try:
        from medpy.metric.binary import hd95
        return hd95(pred, target)
    except ImportError:
        from scipy.ndimage import distance_transform_edt
        
        pred_boundary = pred.astype(bool)
        target_boundary = target.astype(bool)
        
        pred_dist = distance_transform_edt(~pred_boundary)
        target_dist = distance_transform_edt(~target_boundary)
        
        dist_pred_to_target = pred_dist[target_boundary]
        dist_target_to_pred = target_dist[pred_boundary]
        
        if len(dist_pred_to_target) == 0 or len(dist_target_to_pred) == 0:
            return np.inf
        
        all_distances = np.concatenate([dist_pred_to_target, dist_target_to_pred])
        return np.percentile(all_distances, 95)


def compute_metrics(pred: np.ndarray, target: np.ndarray) -> Dict[str, float]:
    """计算所有评估指标
    
    Args:
        pred: 预测二值掩码 (H, W) 或 (N, H, W)
        target: 真实二值掩码 (H, W) 或 (N, H, W)
    
    Returns:
        包含所有指标的字典
    """
    pred = pred.astype(np.float32)
    target = target.astype(np.float32)
    
    if pred.ndim == 2:
        pred = pred[np.newaxis, ...]
        target = target[np.newaxis, ...]
    
    metrics = {
        'DSC': [],
        'IOU': [],
        'Precision': [],
        'Recall': [],
        'Specificity': [],
        'HD95': []
    }
    
    for p, t in zip(pred, target):
        metrics['DSC'].append(dice_coefficient(p, t))
        metrics['IOU'].append(iou_score(p, t))
        metrics['Precision'].append(precision_score(p, t))
        metrics['Recall'].append(recall_score(p, t))
        metrics['Specificity'].append(specificity_score(p, t))
        metrics['HD95'].append(hausdorff_distance_95(p, t))
    
    return {k: float(np.mean(v)) for k, v in metrics.items()}


class DiceLoss(torch.nn.Module):
    """Dice 损失函数"""
    
    def __init__(self, smooth: float = 1e-6):
        super().__init__()
        self.smooth = smooth
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred = torch.sigmoid(pred)
        
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)
        
        intersection = (pred_flat * target_flat).sum()
        dice = (2.0 * intersection + self.smooth) / (
            pred_flat.sum() + target_flat.sum() + self.smooth
        )
        
        return 1 - dice


class BCEDiceLoss(torch.nn.Module):
    """BCE + Dice 组合损失"""
    
    def __init__(self, bce_weight: float = 0.5, dice_weight: float = 0.5):
        super().__init__()
        self.bce = torch.nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.bce_weight * self.bce(pred, target) + self.dice_weight * self.dice(pred, target)
