#!/usr/bin/env python
"""批量运行所有实验"""

import subprocess
import sys
from pathlib import Path


EXPERIMENTS = [
    {'dataset': 'busi', 'model': 'unet', 'gpu': 0},
    {'dataset': 'busuclm', 'model': 'unet', 'gpu': 1},
]


def run_experiment(dataset: str, model: str, gpu: int):
    """运行单个实验"""
    print(f'\n{"="*60}')
    print(f'Running: {model.upper()} on {dataset.upper()} (GPU {gpu})')
    print(f'{"="*60}\n')
    
    script_dir = Path(__file__).parent
    
    if model == 'unet':
        cmd = [
            sys.executable,
            str(script_dir / 'run_unet.py'),
            '--dataset', dataset,
            '--data_root', './data',
            '--output_dir', './results',
            '--epochs', '200',
            '--batch_size', '4',
            '--lr', '0.0001',
            '--patience', '20',
            '--gpu', str(gpu)
        ]
    else:
        raise ValueError(f'Unknown model: {model}')
    
    result = subprocess.run(cmd, cwd=str(script_dir.parent))
    return result.returncode == 0


def main():
    print('Starting all experiments...')
    print(f'Total experiments: {len(EXPERIMENTS)}')
    
    results = {}
    for exp in EXPERIMENTS:
        key = f"{exp['model']}_{exp['dataset']}"
        success = run_experiment(exp['dataset'], exp['model'], exp['gpu'])
        results[key] = 'SUCCESS' if success else 'FAILED'
    
    print('\n' + '=' * 60)
    print('Experiment Summary')
    print('=' * 60)
    for key, status in results.items():
        print(f'{key}: {status}')
    print('=' * 60)


if __name__ == '__main__':
    main()
