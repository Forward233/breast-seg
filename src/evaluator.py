import os
import json
from pathlib import Path
from typing import Dict

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from .metrics import compute_metrics


class Evaluator:
    """测试集评估与预测保存"""
    
    def __init__(
        self,
        model: nn.Module,
        test_loader: DataLoader,
        device: torch.device,
        save_dir: str
    ):
        self.model = model.to(device)
        self.test_loader = test_loader
        self.device = device
        self.save_dir = Path(save_dir)
        
        self.predictions_dir = self.save_dir / 'predictions'
        self.predictions_dir.mkdir(parents=True, exist_ok=True)
    
    @torch.no_grad()
    def evaluate(self, save_predictions: bool = True) -> Dict[str, float]:
        """在测试集上评估并保存预测结果
        
        Args:
            save_predictions: 是否保存预测图片
        
        Returns:
            评估指标字典（平均值）
        """
        self.model.eval()
        
        all_preds = []
        all_targets = []
        all_names = []
        
        for images, masks, names in tqdm(self.test_loader, desc='Evaluating'):
            images = images.to(self.device)
            
            outputs = self.model(images)
            preds = (torch.sigmoid(outputs) > 0.5).float()
            
            preds_np = preds.cpu().numpy()
            masks_np = masks.numpy()
            
            all_preds.append(preds_np)
            all_targets.append(masks_np)
            all_names.extend(names)
            
            if save_predictions:
                for pred, name in zip(preds_np, names):
                    pred_img = (pred.squeeze() * 255).astype(np.uint8)
                    cv2.imwrite(str(self.predictions_dir / name), pred_img)
        
        all_preds = np.concatenate(all_preds, axis=0)
        all_targets = np.concatenate(all_targets, axis=0)
        
        metrics = compute_metrics(all_preds.squeeze(), all_targets.squeeze())
        
        metrics_path = self.save_dir / 'test_metrics.json'
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        print('\n' + '=' * 50)
        print('Test Results (Mean):')
        print('=' * 50)
        for k, v in metrics.items():
            if k == 'HD95' and np.isinf(v):
                print(f'{k}: inf')
            else:
                print(f'{k}: {v:.4f}')
        print('=' * 50)
        
        return metrics
    
    def load_checkpoint(self, checkpoint_path: str):
        """加载模型检查点"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        print(f'Loaded model from {checkpoint_path}')
        return checkpoint.get('epoch', -1), checkpoint.get('best_dice', 0.0)
