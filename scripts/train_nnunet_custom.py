#!/usr/bin/env python
"""自定义nnUNet训练脚本 - 完全按需求实现训练参数"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, Tuple

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import cv2
import numpy as np

# nnUNet架构依赖
# PlainConvUNet来自dynamic_network_architectures包


class SimpleNnUNetDataset(Dataset):
    """简单的nnUNet数据集加载器"""
    
    def __init__(self, images_dir: str, labels_dir: str, image_size: int = 256):
        self.images_dir = Path(images_dir)
        self.labels_dir = Path(labels_dir)
        self.image_size = image_size
        
        self.image_files = sorted([f for f in os.listdir(images_dir) if f.endswith('_0000.png')])
        
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        case_id = img_name.replace('_0000.png', '')
        
        img = cv2.imread(str(self.images_dir / img_name), cv2.IMREAD_GRAYSCALE)
        label = cv2.imread(str(self.labels_dir / f'{case_id}.png'), cv2.IMREAD_GRAYSCALE)
        
        img = cv2.resize(img, (self.image_size, self.image_size))
        label = cv2.resize(label, (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST)
        
        # 归一化到[0,1]
        img = img.astype(np.float32) / 255.0
        label = (label > 0).astype(np.float32)
        
        img = torch.from_numpy(img).unsqueeze(0)
        label = torch.from_numpy(label).unsqueeze(0)
        
        return img, label, case_id


class BCEDiceLoss(nn.Module):
    """BCE + Dice混合损失"""
    
    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
    
    def forward(self, pred, target):
        bce_loss = self.bce(pred, target)
        
        pred_sigmoid = torch.sigmoid(pred)
        smooth = 1e-6
        intersection = (pred_sigmoid * target).sum(dim=(2, 3))
        union = pred_sigmoid.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
        dice = (2.0 * intersection + smooth) / (union + smooth)
        dice_loss = 1.0 - dice.mean()
        
        return bce_loss + dice_loss


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


def dice_coefficient(pred: np.ndarray, target: np.ndarray, smooth: float = 1e-6) -> float:
    """计算Dice系数"""
    intersection = np.sum(pred * target)
    return (2.0 * intersection + smooth) / (np.sum(pred) + np.sum(target) + smooth)


def train_custom_nnunet(
    dataset_id: int,
    nnunet_preprocessed_dir: str,
    output_dir: str,
    epochs: int = 200,
    batch_size: int = 4,
    learning_rate: float = 1e-4,
    patience: int = 20,
    image_size: int = 256,
    device: str = 'cuda:0'
):
    """自定义nnUNet训练"""
    
    # 设置路径
    dataset_name = f'Dataset{dataset_id:03d}_{"BUSI" if dataset_id == 1 else "BUSUCLM"}'
    nnunet_raw_dir = Path(nnunet_preprocessed_dir).parent / 'nnUNet_raw' / dataset_name
    
    save_dir = Path(output_dir) / f'nnUNet_{dataset_name.split("_")[1]}'
    save_dir.mkdir(parents=True, exist_ok=True)
    (save_dir / 'checkpoints').mkdir(exist_ok=True)
    
    # 设置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(save_dir / 'training.log'),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)
    
    logger.info(f'Training {dataset_name}')
    logger.info(f'Parameters: epochs={epochs}, batch_size={batch_size}, lr={learning_rate}, patience={patience}')
    
    # 加载数据
    train_dataset = SimpleNnUNetDataset(
        str(nnunet_raw_dir / 'imagesTr'),
        str(nnunet_raw_dir / 'labelsTr'),
        image_size
    )
    
    # 划分train/val (7:1比例)
    total_size = len(train_dataset)
    val_size = max(1, total_size // 8)
    train_size = total_size - val_size
    
    generator = torch.Generator().manual_seed(42)
    train_subset, val_subset = torch.utils.data.random_split(
        train_dataset, [train_size, val_size], generator=generator
    )
    
    test_dataset = SimpleNnUNetDataset(
        str(nnunet_raw_dir / 'imagesTs'),
        str(nnunet_raw_dir / 'labelsTs'),
        image_size
    )
    
    # 平台兼容性检查
    import platform
    num_workers = 0 if platform.system() == 'Windows' else 4
    
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    
    logger.info(f'Train: {len(train_subset)}, Val: {len(val_subset)}, Test: {len(test_dataset)}')
    
    # 使用nnUNet的PlainConvUNet架构
    from dynamic_network_architectures.architectures.unet import PlainConvUNet
    
    # 定义nnUNet的网络结构（基于预处理配置）
    model = PlainConvUNet(
        input_channels=1,
        n_stages=7,
        features_per_stage=[32, 64, 128, 256, 512, 512, 512],
        conv_op=nn.Conv2d,
        kernel_sizes=[[3, 3]] * 7,
        strides=[[1, 1], [2, 2], [2, 2], [2, 2], [2, 2], [2, 2], [2, 2]],
        n_conv_per_stage=[2, 2, 2, 2, 2, 2, 2],
        num_classes=1,
        n_conv_per_stage_decoder=[2, 2, 2, 2, 2, 2],
        conv_bias=True,
        norm_op=nn.InstanceNorm2d,
        norm_op_kwargs={'eps': 1e-5, 'affine': True},
        dropout_op=None,
        nonlin=nn.LeakyReLU,
        nonlin_kwargs={'inplace': True}
    ).to(device)
    
    logger.info(f'Model: nnUNet PlainConvUNet, Parameters: {sum(p.numel() for p in model.parameters()):,}')
    
    # 训练配置
    criterion = BCEDiceLoss()
    optimizer = Adam(model.parameters(), lr=learning_rate)
    early_stopping = EarlyStopping(patience=patience)
    
    history = {'train_loss': [], 'val_loss': [], 'val_dice': []}
    best_dice = 0.0
    best_epoch = 0
    
    # 训练循环
    for epoch in range(epochs):
        # 训练阶段
        model.train()
        train_loss = 0.0
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs} [Train]')
        for images, masks, _ in pbar:
            images = images.to(device)
            masks = masks.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})
        
        train_loss /= len(train_loader)
        
        # 验证阶段
        model.eval()
        val_loss = 0.0
        val_dice_scores = []
        
        with torch.no_grad():
            for images, masks, _ in val_loader:
                images = images.to(device)
                masks = masks.to(device)
                
                outputs = model(images)
                loss = criterion(outputs, masks)
                val_loss += loss.item()
                
                preds = torch.sigmoid(outputs) > 0.5
                for pred, mask in zip(preds.cpu().numpy(), masks.cpu().numpy()):
                    val_dice_scores.append(dice_coefficient(pred, mask))
        
        val_loss /= len(val_loader)
        val_dice = np.mean(val_dice_scores)
        
        history['train_loss'].append(float(train_loss))
        history['val_loss'].append(float(val_loss))
        history['val_dice'].append(float(val_dice))
        
        logger.info(f'Epoch {epoch+1}/{epochs} - train_loss: {train_loss:.4f}, val_loss: {val_loss:.4f}, val_dice: {val_dice:.4f}')
        
        # 保存最佳模型
        is_best = early_stopping(val_dice)
        if is_best:
            best_dice = val_dice
            best_epoch = epoch
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_dice': val_dice
            }, save_dir / 'checkpoints' / 'best_model.pth')
            logger.info(f'  --> Saved best model (dice: {val_dice:.4f})')
        
        if early_stopping.early_stop:
            logger.info(f'Early stopping triggered at epoch {epoch+1}')
            break
    
    # 保存训练历史
    with open(save_dir / 'history.json', 'w') as f:
        json.dump(history, f, indent=2)
    
    logger.info(f'Training completed. Best epoch: {best_epoch+1}, Best dice: {best_dice:.4f}')
    
    # 测试阶段
    logger.info('Evaluating on test set...')
    checkpoint = torch.load(save_dir / 'checkpoints' / 'best_model.pth', weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    from src.metrics import compute_metrics
    all_preds = []
    all_targets = []
    pred_save_dir = save_dir / 'predictions'
    pred_save_dir.mkdir(exist_ok=True)
    
    with torch.no_grad():
        for images, masks, case_ids in tqdm(test_loader, desc='Testing'):
            images = images.to(device)
            outputs = model(images)
            preds = torch.sigmoid(outputs) > 0.5
            
            for pred, mask, case_id in zip(preds.cpu().numpy(), masks.numpy(), case_ids):
                all_preds.append(pred)
                all_targets.append(mask)
                
                pred_img = (pred[0] * 255).astype(np.uint8)
                cv2.imwrite(str(pred_save_dir / f'{case_id}.png'), pred_img)
    
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    test_metrics = compute_metrics(all_preds, all_targets)
    
    with open(save_dir / 'test_metrics.json', 'w') as f:
        json.dump(test_metrics, f, indent=2)
    
    logger.info(f'Test metrics: {test_metrics}')
    logger.info(f'Results saved to {save_dir}')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Custom nnUNet Training')
    parser.add_argument('--dataset_id', type=int, required=True, help='Dataset ID (1 for BUSI, 2 for BUSUCLM)')
    parser.add_argument('--nnunet_preprocessed', type=str, default='./nnunet_data/nnUNet_preprocessed',
                        help='nnUNet preprocessed directory')
    parser.add_argument('--output_dir', type=str, default='./results', help='Output directory')
    parser.add_argument('--epochs', type=int, default=200, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--patience', type=int, default=20, help='Early stopping patience')
    parser.add_argument('--gpu', type=int, default=0, help='GPU device ID')
    args = parser.parse_args()
    
    device = f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu'
    
    train_custom_nnunet(
        dataset_id=args.dataset_id,
        nnunet_preprocessed_dir=args.nnunet_preprocessed,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        patience=args.patience,
        device=device
    )
