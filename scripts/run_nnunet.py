#!/usr/bin/env python
"""nnUNet 训练与评估脚本

nnUNet 使用命令行方式运行，本脚本封装相关命令。

使用前需要设置环境变量:
export nnUNet_raw="/path/to/nnunet_data/nnUNet_raw"
export nnUNet_preprocessed="/path/to/nnunet_data/nnUNet_preprocessed"
export nnUNet_results="/path/to/nnunet_data/nnUNet_results"
"""

import os
import subprocess
import argparse
from pathlib import Path


def setup_nnunet_env(base_dir: str):
    """设置 nnUNet 环境变量"""
    base = Path(base_dir).absolute()
    
    os.environ['nnUNet_raw'] = str(base / 'nnUNet_raw')
    os.environ['nnUNet_preprocessed'] = str(base / 'nnUNet_preprocessed')
    os.environ['nnUNet_results'] = str(base / 'nnUNet_results')
    
    for key in ['nnUNet_raw', 'nnUNet_preprocessed', 'nnUNet_results']:
        Path(os.environ[key]).mkdir(parents=True, exist_ok=True)
        print(f'{key}: {os.environ[key]}')


def run_nnunet_plan_and_preprocess(dataset_id: int):
    """运行 nnUNet 规划和预处理"""
    cmd = [
        'nnUNetv2_plan_and_preprocess',
        '-d', str(dataset_id),
        '--verify_dataset_integrity'
    ]
    print(f'Running: {" ".join(cmd)}')
    subprocess.run(cmd, check=True)


def run_nnunet_train(dataset_id: int, fold: int = 0, config: str = '2d', num_epochs: int = 200):
    """运行 nnUNet 训练
    
    Args:
        dataset_id: 数据集 ID
        fold: 交叉验证折数 (0-4 或 'all')
        config: 配置 ('2d', '3d_fullres', '3d_lowres')
        num_epochs: 训练轮数
    """
    cmd = [
        'nnUNetv2_train',
        str(dataset_id),
        config,
        str(fold),
        '--npz',
        '--num_epochs', str(num_epochs)
    ]
    print(f'Running: {" ".join(cmd)}')
    subprocess.run(cmd, check=True)


def run_nnunet_predict(dataset_id: int, input_dir: str, output_dir: str, config: str = '2d', fold: int = 0):
    """运行 nnUNet 预测"""
    cmd = [
        'nnUNetv2_predict',
        '-i', input_dir,
        '-o', output_dir,
        '-d', str(dataset_id),
        '-c', config,
        '-f', str(fold)
    ]
    print(f'Running: {" ".join(cmd)}')
    subprocess.run(cmd, check=True)


def main():
    parser = argparse.ArgumentParser(description='nnUNet training pipeline')
    parser.add_argument('--action', type=str, required=True,
                        choices=['preprocess', 'train', 'predict', 'all'],
                        help='Action to perform')
    parser.add_argument('--dataset_id', type=int, required=True,
                        help='Dataset ID (1 for BUSI, 2 for BUSUCLM)')
    parser.add_argument('--nnunet_base', type=str, default='./nnunet_data',
                        help='nnUNet base directory')
    parser.add_argument('--fold', type=int, default=0,
                        help='Fold for training')
    parser.add_argument('--config', type=str, default='2d',
                        help='nnUNet configuration')
    parser.add_argument('--num_epochs', type=int, default=200,
                        help='Number of training epochs')
    args = parser.parse_args()
    
    setup_nnunet_env(args.nnunet_base)
    
    if args.action in ['preprocess', 'all']:
        print('\n=== Preprocessing ===')
        run_nnunet_plan_and_preprocess(args.dataset_id)
    
    if args.action in ['train', 'all']:
        print('\n=== Training ===')
        run_nnunet_train(args.dataset_id, args.fold, args.config, args.num_epochs)
    
    if args.action in ['predict', 'all']:
        print('\n=== Predicting ===')
        raw_dir = Path(os.environ['nnUNet_raw'])
        dataset_name = f'Dataset{args.dataset_id:03d}_*'
        dataset_dirs = list(raw_dir.glob(dataset_name))
        if dataset_dirs:
            input_dir = str(dataset_dirs[0] / 'imagesTs')
            output_dir = f'./results/nnUNet_Dataset{args.dataset_id:03d}/predictions'
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            run_nnunet_predict(args.dataset_id, input_dir, output_dir, args.config, args.fold)


if __name__ == '__main__':
    main()
