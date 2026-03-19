import os
import json
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from tqdm import tqdm

from .metrics import BCEDiceLoss, compute_metrics


class EarlyStopping:
    """早停机制"""
    
    def __init__(self, patience: int = 20, min_delta: float = 0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
    
    def __call__(self, val_score: float) -> bool:
        if self.best_score is None:
            self.best_score = val_score
            return True
        
        if val_score > self.best_score + self.min_delta:
            self.best_score = val_score
            self.counter = 0
            return True
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
            return False


class Trainer:
    """训练器"""
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: torch.device,
        save_dir: str,
        learning_rate: float = 1e-4,
        epochs: int = 200,
        patience: int = 20
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.save_dir = Path(save_dir)
        self.epochs = epochs
        
        self.save_dir.mkdir(parents=True, exist_ok=True)
        (self.save_dir / 'checkpoints').mkdir(exist_ok=True)
        
        self.criterion = BCEDiceLoss()
        self.optimizer = Adam(model.parameters(), lr=learning_rate)
        self.early_stopping = EarlyStopping(patience=patience)
        
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'val_dice': []
        }
        self.best_dice = 0.0
        self.best_epoch = 0
        
        self._setup_logging()
    
    def _setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.save_dir / 'training.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def train_epoch(self) -> float:
        """训练一个 epoch"""
        self.model.train()
        total_loss = 0.0
        
        pbar = tqdm(self.train_loader, desc='Training')
        for images, masks, _ in pbar:
            images = images.to(self.device)
            masks = masks.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, masks)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        return total_loss / len(self.train_loader)
    
    @torch.no_grad()
    def validate(self) -> Tuple[float, float]:
        """验证"""
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_targets = []
        
        for images, masks, _ in tqdm(self.val_loader, desc='Validating'):
            images = images.to(self.device)
            masks = masks.to(self.device)
            
            outputs = self.model(images)
            loss = self.criterion(outputs, masks)
            total_loss += loss.item()
            
            preds = (torch.sigmoid(outputs) > 0.5).float()
            all_preds.append(preds.cpu().numpy())
            all_targets.append(masks.cpu().numpy())
        
        import numpy as np
        all_preds = np.concatenate(all_preds, axis=0)
        all_targets = np.concatenate(all_targets, axis=0)
        
        metrics = compute_metrics(all_preds.squeeze(), all_targets.squeeze())
        
        avg_loss = total_loss / len(self.val_loader)
        val_dice = metrics['DSC']
        
        return avg_loss, val_dice
    
    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """保存检查点"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_dice': self.best_dice,
            'history': self.history
        }
        
        if is_best:
            path = self.save_dir / 'checkpoints' / 'best_model.pth'
            torch.save(checkpoint, path)
            self.logger.info(f'Saved best model at epoch {epoch} with dice {self.best_dice:.4f}')
    
    def train(self) -> Dict:
        """完整训练流程"""
        self.logger.info(f'Starting training for {self.epochs} epochs')
        self.logger.info(f'Device: {self.device}')
        self.logger.info(f'Model params: {self.model.get_num_params():,}')
        
        for epoch in range(1, self.epochs + 1):
            self.logger.info(f'\nEpoch {epoch}/{self.epochs}')
            
            train_loss = self.train_epoch()
            val_loss, val_dice = self.validate()
            
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['val_dice'].append(val_dice)
            
            self.logger.info(
                f'Train Loss: {train_loss:.4f} | '
                f'Val Loss: {val_loss:.4f} | '
                f'Val Dice: {val_dice:.4f}'
            )
            
            is_best = self.early_stopping(val_dice)
            if is_best:
                self.best_dice = val_dice
                self.best_epoch = epoch
                self.save_checkpoint(epoch, is_best=True)
            
            if self.early_stopping.early_stop:
                self.logger.info(f'Early stopping at epoch {epoch}')
                break
        
        self.logger.info(f'\nTraining completed!')
        self.logger.info(f'Best Dice: {self.best_dice:.4f} at epoch {self.best_epoch}')
        
        history_serializable = {
            k: [float(v) for v in vals] for k, vals in self.history.items()
        }
        with open(self.save_dir / 'history.json', 'w') as f:
            json.dump(history_serializable, f, indent=2)
        
        return self.history
    
    def load_best_model(self):
        """加载最佳模型"""
        checkpoint_path = self.save_dir / 'checkpoints' / 'best_model.pth'
        if checkpoint_path.exists():
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.logger.info(f'Loaded best model from epoch {checkpoint["epoch"]}')
        else:
            self.logger.warning('No checkpoint found!')
