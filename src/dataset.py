import os
from pathlib import Path
from typing import Tuple, List, Optional

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class BreastUltrasoundDataset(Dataset):
    """乳腺超声图像分割数据集"""
    
    def __init__(
        self,
        images_dir: str,
        masks_dir: str,
        image_size: int = 256,
        normalize: bool = True
    ):
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        self.image_size = image_size
        self.normalize = normalize
        
        self.image_files = sorted([
            f for f in os.listdir(self.images_dir) 
            if f.endswith(('.png', '.jpg', '.jpeg'))
        ])
        
        self._validate_pairs()
    
    def _validate_pairs(self):
        """验证图像和掩码配对"""
        for img_file in self.image_files:
            mask_path = self.masks_dir / img_file
            if not mask_path.exists():
                raise FileNotFoundError(f"Mask not found for {img_file}")
    
    def __len__(self) -> int:
        return len(self.image_files)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, str]:
        img_name = self.image_files[idx]
        
        img_path = self.images_dir / img_name
        mask_path = self.masks_dir / img_name
        
        image = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        
        image = cv2.resize(image, (self.image_size, self.image_size))
        mask = cv2.resize(mask, (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST)
        
        if self.normalize:
            image = image.astype(np.float32) / 255.0
        
        mask = (mask > 127).astype(np.float32)
        
        image = torch.from_numpy(image).unsqueeze(0)
        mask = torch.from_numpy(mask).unsqueeze(0)
        
        return image, mask, img_name


def create_dataloaders(
    dataset_name: str,
    data_root: str,
    batch_size: int = 4,
    image_size: int = 256,
    num_workers: int = 4,
    val_split_ratio: float = 0.1
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """创建训练、验证、测试数据加载器
    
    Args:
        dataset_name: 'busi' 或 'busuclm'
        data_root: 数据根目录
        batch_size: 批大小
        image_size: 图像尺寸
        num_workers: 数据加载线程数
        val_split_ratio: 若无验证集目录，从训练集划分的比例
    
    Returns:
        train_loader, val_loader, test_loader
    """
    from torch.utils.data import random_split
    
    data_root = Path(data_root)
    
    if dataset_name.lower() == 'busi':
        base_path = data_root / 'Dataset_BUSI'
    elif dataset_name.lower() == 'busuclm':
        base_path = data_root / 'BUS-UCLM'
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    val_images_dir = base_path / 'images' / 'imagesVal'
    has_val_split = val_images_dir.exists() and len(list(val_images_dir.glob('*.png'))) > 0
    
    train_images_dir = base_path / 'images' / 'imagesTr'
    train_masks_dir = base_path / 'masks' / 'masksTr'
    test_images_dir = base_path / 'images' / 'imagesTs'
    test_masks_dir = base_path / 'masks' / 'masksTs'
    
    if has_val_split:
        val_images_dir = base_path / 'images' / 'imagesVal'
        val_masks_dir = base_path / 'masks' / 'masksVal'
        
        train_dataset = BreastUltrasoundDataset(
            str(train_images_dir), str(train_masks_dir), image_size
        )
        val_dataset = BreastUltrasoundDataset(
            str(val_images_dir), str(val_masks_dir), image_size
        )
    else:
        full_train_dataset = BreastUltrasoundDataset(
            str(train_images_dir), str(train_masks_dir), image_size
        )
        
        total_size = len(full_train_dataset)
        val_size = int(total_size * val_split_ratio)
        train_size = total_size - val_size
        
        generator = torch.Generator().manual_seed(42)
        train_dataset, val_dataset = random_split(
            full_train_dataset, [train_size, val_size], generator=generator
        )
    
    test_dataset = BreastUltrasoundDataset(
        str(test_images_dir), str(test_masks_dir), image_size
    )
    
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    
    return train_loader, val_loader, test_loader


def get_dataset_info(data_root: str) -> dict:
    """获取数据集信息"""
    data_root = Path(data_root)
    info = {}
    
    for ds_name, ds_path in [('busi', 'Dataset_BUSI'), ('busuclm', 'BUS-UCLM')]:
        base = data_root / ds_path / 'images'
        counts = {}
        for split, subdir in [('train', 'imagesTr'), ('val', 'imagesVal'), ('test', 'imagesTs')]:
            split_path = base / subdir
            if split_path.exists():
                counts[split] = len(list(split_path.glob('*.png')))
            else:
                counts[split] = 0
        info[ds_name] = counts
    
    return info
