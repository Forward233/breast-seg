#!/usr/bin/env python
"""UNet 训练与评估入口脚本"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from src.dataset import create_dataloaders, get_dataset_info
from src.models.unet import UNet
from src.trainer import Trainer
from src.evaluator import Evaluator


def parse_args():
    parser = argparse.ArgumentParser(description='Breast Ultrasound Segmentation with UNet')
    parser.add_argument('--dataset', type=str, required=True, choices=['busi', 'busuclm'],
                        help='Dataset name: busi or busuclm')
    parser.add_argument('--data_root', type=str, default='./data',
                        help='Path to data directory')
    parser.add_argument('--output_dir', type=str, default='./results',
                        help='Output directory')
    parser.add_argument('--epochs', type=int, default=200,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--patience', type=int, default=20,
                        help='Early stopping patience')
    parser.add_argument('--image_size', type=int, default=256,
                        help='Input image size')
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU device ID')
    parser.add_argument('--eval_only', action='store_true',
                        help='Only run evaluation')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to checkpoint for evaluation')
    return parser.parse_args()


def main():
    args = parse_args()
    
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    save_dir = Path(args.output_dir) / f'UNet_{args.dataset.upper()}'
    save_dir.mkdir(parents=True, exist_ok=True)
    
    print(f'\n{"="*50}')
    print(f'Dataset: {args.dataset.upper()}')
    print(f'Output: {save_dir}')
    print(f'{"="*50}\n')
    
    print('Loading data...')
    info = get_dataset_info(args.data_root)
    print(f'Dataset info: {info}')
    
    import platform
    num_workers = 0 if platform.system() == 'Windows' else 4
    
    train_loader, val_loader, test_loader = create_dataloaders(
        dataset_name=args.dataset,
        data_root=args.data_root,
        batch_size=args.batch_size,
        image_size=args.image_size,
        num_workers=num_workers
    )
    
    print(f'Train samples: {len(train_loader.dataset)}')
    print(f'Val samples: {len(val_loader.dataset)}')
    print(f'Test samples: {len(test_loader.dataset)}')
    
    model = UNet(in_channels=1, out_channels=1, bilinear=True)
    print(f'Model parameters: {model.get_num_params():,}')
    
    if args.eval_only:
        checkpoint_path = args.checkpoint or (save_dir / 'checkpoints' / 'best_model.pth')
        evaluator = Evaluator(model, test_loader, device, save_dir)
        evaluator.load_checkpoint(str(checkpoint_path))
        evaluator.evaluate(save_predictions=True)
    else:
        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            save_dir=str(save_dir),
            learning_rate=args.lr,
            epochs=args.epochs,
            patience=args.patience
        )
        
        trainer.train()
        
        print('\nEvaluating on test set...')
        trainer.load_best_model()
        evaluator = Evaluator(trainer.model, test_loader, device, str(save_dir))
        evaluator.evaluate(save_predictions=True)
    
    print('\nDone!')


if __name__ == '__main__':
    main()
