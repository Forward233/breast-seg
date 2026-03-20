# 乳腺超声图像分割项目

基于深度学习的乳腺超声图像分割实验，支持 U-Net 和 nnUNet 两种模型。

## 项目结构

```
breast-seg/
├── configs/
│   └── default.yaml          # 配置文件
├── data/
│   ├── Dataset_BUSI/         # BUSI 数据集
│   └── BUS-UCLM/             # BUS-UCLM 数据集
├── src/
│   ├── dataset.py            # 数据加载
│   ├── metrics.py            # 评估指标
│   ├── trainer.py            # 训练器
│   ├── evaluator.py          # 评估器
│   └── models/
│       └── unet.py           # U-Net 模型
├── scripts/
│   ├── run_unet.py              # UNet 训练入口
│   ├── train_nnunet_custom.py   # nnUNet 自定义训练（完全对齐需求参数）
│   ├── run_nnunet.py            # nnUNet 官方训练入口
│   ├── convert_to_nnunet.py     # 数据格式转换
│   └── run_all_experiments.py
├── results/                   # 输出目录
└── requirements.txt
```

## 环境配置

```bash
pip install -r requirements.txt
```

## 训练参数

| 参数 | 值 |
|------|-----|
| 输入尺寸 | 256 × 256 |
| 像素归一化 | [0, 1] |
| Epochs | 200 |
| 早停 patience | 20 |
| 优化器 | Adam |
| 学习率 | 1e-4 |
| Batch Size | 4 |

## 评估指标

- DSC (Dice Similarity Coefficient)
- IOU (Intersection over Union)
- Precision
- Recall
- Specificity
- HD95 (95% Hausdorff Distance)

## 使用方法

### 1. U-Net 训练

```bash
# BUSI 数据集
python scripts/run_unet.py --dataset busi --gpu 0

# BUS-UCLM 数据集
python scripts/run_unet.py --dataset busuclm --gpu 1
```

### 2. nnUNet 训练

**安装依赖**：
```bash
pip install nnunetv2 dynamic-network-architectures
```

**完整步骤**：
```bash
# 1. 转换数据格式（生成nnUNet格式数据）
python scripts/convert_to_nnunet.py --data_root ./data

# 2. 训练（使用自定义脚本，完全对齐需求参数）
# BUSI 数据集
CUDA_VISIBLE_DEVICES=0 python scripts/train_nnunet_custom.py \
  --dataset_id 1 \
  --epochs 200 \
  --batch_size 4 \
  --lr 1e-4 \
  --patience 20 \
  --gpu 0

# BUS-UCLM 数据集
CUDA_VISIBLE_DEVICES=0 python scripts/train_nnunet_custom.py \
  --dataset_id 2 \
  --epochs 200 \
  --batch_size 4 \
  --lr 1e-4 \
  --patience 20 \
  --gpu 0
```

**后台运行**：
```bash
CUDA_VISIBLE_DEVICES=0 nohup python scripts/train_nnunet_custom.py \
  --dataset_id 1 --epochs 200 --batch_size 4 --lr 1e-4 --patience 20 \
  > nnunet_busi.log 2>&1 &
```

**自定义脚本特性**：
- 使用 nnUNet PlainConvUNet 架构
- 完全对齐需求参数（Adam优化器、lr=1e-4、早停等）
- 自动划分 train/val (7:1)，test 单独处理
- 输出与 UNet 格式一致

### 3. 批量运行

```bash
python scripts/run_all_experiments.py
```

## 输出结构

```
results/
├── UNet_BUSI/
│   ├── checkpoints/best_model.pth
│   ├── predictions/
│   ├── test_metrics.json
│   └── training.log
├── UNet_BUSUCLM/
├── nnUNet_BUSI/
└── nnUNet_BUSUCLM/
```
