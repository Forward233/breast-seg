#!/usr/bin/env python
"""将数据集转换为 nnUNet 格式"""

import os
import json
import shutil
from pathlib import Path
from typing import List

import cv2
import numpy as np


def convert_to_nnunet_format(
    src_data_root: str,
    nnunet_raw_dir: str,
    dataset_name: str,
    dataset_id: int
):
    """将数据集转换为 nnUNet 格式
    
    nnUNet 数据格式:
    nnUNet_raw/
    └── Dataset{ID}_{Name}/
        ├── dataset.json
        ├── imagesTr/
        │   └── {case_id}_0000.png
        ├── labelsTr/
        │   └── {case_id}.png
        ├── imagesTs/
        │   └── {case_id}_0000.png
        └── labelsTs/
            └── {case_id}.png
    """
    src_root = Path(src_data_root)
    nnunet_dir = Path(nnunet_raw_dir) / f'Dataset{dataset_id:03d}_{dataset_name}'
    
    nnunet_dir.mkdir(parents=True, exist_ok=True)
    (nnunet_dir / 'imagesTr').mkdir(exist_ok=True)
    (nnunet_dir / 'labelsTr').mkdir(exist_ok=True)
    (nnunet_dir / 'imagesTs').mkdir(exist_ok=True)
    (nnunet_dir / 'labelsTs').mkdir(exist_ok=True)
    
    train_cases = []
    test_cases = []
    
    src_images_tr = src_root / 'images' / 'imagesTr'
    src_masks_tr = src_root / 'masks' / 'masksTr'
    
    for img_file in sorted(src_images_tr.glob('*.png')):
        case_id = img_file.stem
        
        img = cv2.imread(str(img_file), cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(str(src_masks_tr / img_file.name), cv2.IMREAD_GRAYSCALE)
        
        cv2.imwrite(str(nnunet_dir / 'imagesTr' / f'{case_id}_0000.png'), img)
        
        mask_binary = (mask > 127).astype(np.uint8)
        cv2.imwrite(str(nnunet_dir / 'labelsTr' / f'{case_id}.png'), mask_binary)
        
        train_cases.append(case_id)
    
    src_images_ts = src_root / 'images' / 'imagesTs'
    src_masks_ts = src_root / 'masks' / 'masksTs'
    
    for img_file in sorted(src_images_ts.glob('*.png')):
        case_id = img_file.stem
        
        img = cv2.imread(str(img_file), cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(str(src_masks_ts / img_file.name), cv2.IMREAD_GRAYSCALE)
        
        cv2.imwrite(str(nnunet_dir / 'imagesTs' / f'{case_id}_0000.png'), img)
        
        mask_binary = (mask > 127).astype(np.uint8)
        cv2.imwrite(str(nnunet_dir / 'labelsTs' / f'{case_id}.png'), mask_binary)
        
        test_cases.append(case_id)
    
    dataset_json = {
        "channel_names": {
            "0": "ultrasound"
        },
        "labels": {
            "background": 0,
            "lesion": 1
        },
        "numTraining": len(train_cases),
        "file_ending": ".png",
        "name": dataset_name,
        "description": f"Breast ultrasound segmentation - {dataset_name}",
        "reference": "",
        "licence": "",
        "release": "1.0"
    }
    
    with open(nnunet_dir / 'dataset.json', 'w') as f:
        json.dump(dataset_json, f, indent=2)
    
    print(f'Converted {dataset_name}:')
    print(f'  Training cases: {len(train_cases)}')
    print(f'  Test cases: {len(test_cases)}')
    print(f'  Output: {nnunet_dir}')
    
    return str(nnunet_dir)


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Convert to nnUNet format')
    parser.add_argument('--data_root', type=str, default='./data',
                        help='Source data root')
    parser.add_argument('--output_dir', type=str, default='./nnunet_data/nnUNet_raw',
                        help='nnUNet raw data directory')
    args = parser.parse_args()
    
    datasets = [
        ('Dataset_BUSI', 'BUSI', 1),
        ('BUS-UCLM', 'BUSUCLM', 2),
    ]
    
    for src_name, nnunet_name, dataset_id in datasets:
        src_path = Path(args.data_root) / src_name
        if src_path.exists():
            convert_to_nnunet_format(
                str(src_path),
                args.output_dir,
                nnunet_name,
                dataset_id
            )
        else:
            print(f'Warning: {src_path} not found, skipping')


if __name__ == '__main__':
    main()
