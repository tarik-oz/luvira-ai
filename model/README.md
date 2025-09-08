# Deep Learning Model Module

This module contains the deep learning components for hair segmentation using Attention U-Net architecture. It provides training scripts, model architectures, data loaders, and inference capabilities.

## 📋 Table of Contents

- [Setup](#setup)
- [Model Architecture](#model-architecture)
- [Training a New Model](#training-a-new-model)
- [Running Inference](#running-inference)
- [CLI Tools](#cli-tools)
- [Data Preparation](#data-preparation)
- [Model Configuration](#model-configuration)
- [Troubleshooting](#troubleshooting)

## 🚀 Setup

### Prerequisites

- Python 3.10+
- CUDA-compatible GPU (optional, CPU training supported)

### 1. Create Virtual Environment

```bash
# Navigate to project root
cd deep-learning-hair-segmentation

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate
```

### 2. Install Dependencies

```bash
# Install all project dependencies
pip install -r requirements.txt

# Or install minimal dependencies for model only
pip install torch torchvision opencv-python numpy matplotlib scikit-learn tqdm
```

### 3. Verify Installation

```bash
# Test model import
python -c "from model.models.attention_unet_model import AttentionUNet; print('Model import successful')"

# Check CUDA availability (optional)
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

## 🧠 Model Architecture

### Attention U-Net

The project uses an Attention U-Net architecture optimized for hair segmentation:

```
Input (3, 256, 256) → Encoder → Bottleneck → Decoder → Output (1, 256, 256)
                    ↓                      ↑
                Attention Gates ──────────┘
```

**Key Features:**

- **Attention Mechanism**: Focuses on relevant hair regions
- **Skip Connections**: Preserves fine details
- **Multi-scale Processing**: Handles various hair textures
- **Optimized for Hair**: Trained specifically on hair segmentation data

### Available Models

- `AttentionUNet`: Main segmentation model with attention gates
- `UNet`: Standard U-Net implementation (fallback)

## 🏋️ Training a New Model

### 1. Prepare Your Dataset

```bash
# Organize your data in this structure:
dataset/
├── images/           # Original images
│   ├── img001.jpg
│   ├── img002.jpg
│   └── ...
└── masks/           # Hair masks (white=hair, black=background)
    ├── img001.png
    ├── img002.png
    └── ...
```

### 2. Configure Training

Edit `model/config.py` or create environment variables:

```python
# Basic configuration
DATA_CONFIG = {
    "dataset_path": "/path/to/your/dataset",
    "batch_size": 8,
    "image_size": 256,
    "train_split": 0.8
}

MODEL_CONFIG = {
    "model_type": "attention_unet",
    "input_channels": 3,
    "output_channels": 1,
    "learning_rate": 0.001
}
```

### 3. Start Training

```bash
# Basic training with default config
python -m model.training.train_model

# Training with custom dataset path
python -m model.training.train_model --dataset-path /path/to/dataset

# Training with custom parameters
python -m model.training.train_model \
    --dataset-path /path/to/dataset \
    --batch-size 16 \
    --epochs 100 \
    --learning-rate 0.0001 \
    --model-type attention_unet
```

### 4. Monitor Training

```bash
# Training will create timestamped output directory:
model/trained_models/2024-01-15_14-30-45/
├── best_model.pth       # Best model weights
├── final_model.pth      # Final model weights
├── config.json          # Training configuration
├── metrics.json         # Training metrics
└── training.log         # Detailed logs
```

### 5. Advanced Training Options

```bash
# Resume from checkpoint
python -m model.training.train_model --resume model/trained_models/2024-01-15_14-30-45/

# Use different data loader
python -m model.training.train_model --data-loader lazy  # or traditional

# Enable data augmentation
python -m model.training.train_model --augmentation

# GPU training
python -m model.training.train_model --device cuda

# Multi-GPU training
python -m model.training.train_model --device cuda --multi-gpu
```

## 🔮 Running Inference

### 1. Single Image Prediction

```bash
# Predict on single image
python -m model.inference.predict \
    --model-path model/trained_models/2024-01-15_14-30-45/best_model.pth \
    --input-image /path/to/image.jpg \
    --output-mask /path/to/output_mask.png

# Predict with visualization
python -m model.inference.predict \
    --model-path model/trained_models/2024-01-15_14-30-45/best_model.pth \
    --input-image /path/to/image.jpg \
    --output-mask /path/to/output_mask.png \
    --visualize \
    --output-overlay /path/to/overlay.png
```

### 2. Batch Prediction

```bash
# Predict on directory of images
python -m model.inference.predict \
    --model-path model/trained_models/2024-01-15_14-30-45/best_model.pth \
    --input-dir /path/to/images/ \
    --output-dir /path/to/masks/

# Batch prediction with custom threshold
python -m model.inference.predict \
    --model-path model/trained_models/2024-01-15_14-30-45/best_model.pth \
    --input-dir /path/to/images/ \
    --output-dir /path/to/masks/ \
    --threshold 0.7
```

### 3. Python API Usage

```python
from model.inference.predictor import create_predictor
from model.training.trainer import create_trainer

# Load trained model
trainer = create_trainer()
model, config = trainer.load_trained_model("path/to/best_model.pth")

# Create predictor
predictor = create_predictor(model)

# Single prediction
original_img, predicted_mask, overlay_img = predictor.predict("image.jpg")

# The mask is a numpy array with values 0-1
print(f"Mask shape: {predicted_mask.shape}")
print(f"Hair pixels: {(predicted_mask > 0.5).sum()}")
```

## 🛠️ CLI Tools

### Model Information

```bash
# Get model architecture summary
python -c "from model.models.attention_unet_model import AttentionUNet; \
          model = AttentionUNet(); \
          print(f'Parameters: {sum(p.numel() for p in model.parameters()):,}')"

# Validate model file
python -m model.training.trainer --validate-model path/to/model.pth
```

### Data Validation

```bash
# Validate dataset structure
python -m model.utils.validators --validate-dataset /path/to/dataset

# Check data loader
python -c "from model.data_loader.factory_data_loader import create_data_loader; \
          loader = create_data_loader('traditional', '/path/to/dataset'); \
          print(f'Dataset size: {len(loader.dataset)}')"
```

### Training Utilities

```bash
# Create timestamp for training run
python -c "from model.utils.data_timestamp import get_timestamp_str; \
          print(f'Timestamp: {get_timestamp_str()}')"

# Test data augmentation
python -c "from model.utils.augmentation import get_train_transforms; \
          transforms = get_train_transforms(); \
          print('Augmentation pipeline created')"
```

## 📊 Data Preparation

### Dataset Requirements

- **Images**: JPG/PNG format, any resolution (will be resized to 256x256)
- **Masks**: PNG format, binary masks (0=background, 255=hair)
- **Naming**: Corresponding images and masks must have same filename

### Data Quality Guidelines

```bash
# Check image-mask pairs
python -c "
import os
from pathlib import Path

dataset_path = '/path/to/dataset'
img_dir = Path(dataset_path) / 'images'
mask_dir = Path(dataset_path) / 'masks'

img_files = {f.stem for f in img_dir.glob('*')}
mask_files = {f.stem for f in mask_dir.glob('*')}

print(f'Images: {len(img_files)}')
print(f'Masks: {len(mask_files)}')
print(f'Matched pairs: {len(img_files & mask_files)}')
print(f'Missing masks: {img_files - mask_files}')
print(f'Missing images: {mask_files - img_files}')
"
```

### Data Augmentation

The training pipeline includes automatic augmentation:

- Random horizontal flips
- Random rotations (±15°)
- Random brightness/contrast
- Random blur
- Elastic transformations

## ⚙️ Model Configuration

### Configuration File Structure

```json
{
  "model_config": {
    "model_type": "attention_unet",
    "input_channels": 3,
    "output_channels": 1,
    "input_shape": [3, 256, 256]
  },
  "training_config": {
    "batch_size": 8,
    "learning_rate": 0.001,
    "epochs": 100,
    "optimizer": "adam"
  },
  "data_config": {
    "dataset_path": "/path/to/dataset",
    "train_split": 0.8,
    "validation_split": 0.2
  }
}
```

### Environment Variables

```bash
# Model configuration
export MODEL_TYPE=attention_unet
export INPUT_CHANNELS=3
export OUTPUT_CHANNELS=1
export BATCH_SIZE=8
export LEARNING_RATE=0.001

# Data configuration
export DATASET_PATH=/path/to/dataset
export TRAIN_SPLIT=0.8
export IMAGE_SIZE=256

# Training configuration
export EPOCHS=100
export DEVICE=auto  # auto, cpu, cuda
export DATA_LOADER=traditional  # traditional, lazy
```

## 🔧 Troubleshooting

### Common Issues

#### 1. CUDA Out of Memory

```bash
# Reduce batch size
python -m model.training.train_model --batch-size 4

# Use gradient accumulation
python -m model.training.train_model --batch-size 4 --gradient-accumulation 2

# Use CPU training
python -m model.training.train_model --device cpu
```

#### 2. Dataset Loading Errors

```bash
# Check dataset structure
ls dataset/images/ | head -5
ls dataset/masks/ | head -5

# Validate file formats
python -c "
from PIL import Image
import os

# Check random image
img_path = 'dataset/images/img001.jpg'
mask_path = 'dataset/masks/img001.png'

img = Image.open(img_path)
mask = Image.open(mask_path)

print(f'Image: {img.size}, {img.mode}')
print(f'Mask: {mask.size}, {mask.mode}')
"
```

#### 3. Model Loading Issues

```bash
# Check model file
python -c "
import torch

model_path = 'model/trained_models/xxx/best_model.pth'
checkpoint = torch.load(model_path, map_location='cpu')

print(f'Keys in checkpoint: {list(checkpoint.keys())}')
if 'model_state_dict' in checkpoint:
    print(f'Model parameters: {len(checkpoint[\"model_state_dict\"])}')
"
```

#### 4. Training Not Converging

```bash
# Check learning rate
python -m model.training.train_model --learning-rate 0.0001

# Enable learning rate scheduling
python -m model.training.train_model --lr-scheduler

# Increase training data or augmentation
python -m model.training.train_model --augmentation --epochs 200
```

### Performance Tips

1. **Fast Training**: Use `lazy` data loader for large datasets
2. **Memory Optimization**: Reduce batch size, use gradient accumulation
3. **Quality Training**: Use data augmentation, longer training
4. **GPU Utilization**: Monitor with `nvidia-smi` during training

### Logs and Debugging

```bash
# Check training logs
tail -f model/trained_models/2024-01-15_14-30-45/training.log

# Validate training metrics
python -c "
import json
with open('model/trained_models/xxx/metrics.json') as f:
    metrics = json.load(f)
print(f'Best validation loss: {min(metrics[\"val_loss\"])}')
print(f'Best validation IoU: {max(metrics[\"val_iou\"])}')
"
```

## 📈 Model Evaluation

### Metrics

The training automatically tracks:

- **Loss**: Binary cross-entropy + Dice loss
- **IoU (Intersection over Union)**: Segmentation accuracy
- **Dice Coefficient**: Overlap measurement
- **Pixel Accuracy**: Overall accuracy

### Custom Evaluation

```bash
# Evaluate on test set
python -c "
from model.training.metrics import calculate_iou, calculate_dice
import numpy as np

# Your evaluation code here
predicted_mask = np.load('predicted_mask.npy')
ground_truth = np.load('ground_truth.npy')

iou = calculate_iou(predicted_mask, ground_truth)
dice = calculate_dice(predicted_mask, ground_truth)

print(f'IoU: {iou:.4f}')
print(f'Dice: {dice:.4f}')
"
```

---

For questions or issues, please refer to the main project [README](../README.md) or open an issue in the repository.
