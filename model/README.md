# Deep Learning Model

## 1. Overview

This module contains the deep learning components for hair segmentation using Attention U-Net architecture. It provides training scripts, model architectures, data loaders, and inference capabilities for accurate hair mask generation.

## 2. Tech Stack

- **PyTorch** - Deep learning framework
- **OpenCV** - Image processing and augmentation
- **NumPy** - Numerical computing
- **Matplotlib** - Visualization and training plots
- **scikit-learn** - Machine learning utilities
- **tqdm** - Progress bars for training

## 🚀 Getting Started (Local Development)

This guide will get you a local copy of the model module up and running.

### Prerequisites

- Python 3.10+
- A virtual environment (e.g., `venv`) is recommended
- CUDA-compatible GPU (optional, CPU training supported)

### Installation & Model Setup

1. **Create and activate a virtual environment from the project root:**

   ```bash
   python -m venv venv
   source venv/bin/activate
   ```

2. **Install dependencies from the root requirements.txt:**

   ```bash
   pip install -r requirements.txt
   ```

3. **Install PyTorch (choose CPU or GPU version):**

   ```bash
   # For CPU
   pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cpu

   # For GPU (CUDA 11.8)
   pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu118
   ```

4. **Download the pre-trained model:**

   ```bash
   # Download from Hugging Face
   # Visit: https://huggingface.co/tarik-oz/luviraai-hair-segmentation
   # Download luviraai-model.zip and extract the contents

   # Place the model files in:
   mkdir -p model/trained_models/
   # Copy best_model.pth and config.json to model/trained_models/
   ```

### Running Model Training

```bash
# Basic training with default configuration
python -m model.training.train_model
```

> **Note**: For training configuration and parameter adjustments, see `model/config.py` to modify default values. The training includes checkpoint system, data augmentations, and supports Kaggle environments using the `kaggle/` folder.

### Running Inference

```bash
# Single image prediction
python -m model.inference.predict \
    --model model/trained_models/best_model.pth \
    --images /path/to/image.jpg \
```

> **Note**: For detailed command-line arguments and default parameters, see `model/inference/predict.py` for all available options.

## 🎯 Key Features

- **Attention U-Net Architecture**: Advanced segmentation model with attention gates for precise hair detection
- **Kaggle Integration**: Pre-configured for Kaggle notebook environments with dataset mounting
- **Data Augmentation**: Comprehensive augmentation pipeline including rotations, brightness, and elastic transformations
- **Multi-scale Training**: Handles various image resolutions and hair textures effectively
- **Automated Metrics**: Built-in IoU, Dice coefficient, and pixel accuracy tracking during training

## 🧠 Model Architecture

The project uses an Attention U-Net architecture optimized for hair segmentation:

- **Attention Mechanism**: Focuses on relevant hair regions
- **Skip Connections**: Preserves fine details
- **Multi-scale Processing**: Handles various hair textures
- **Optimized for Hair**: Trained specifically on hair segmentation data

## 📊 Dataset Preparation

For training your own model, organize data in this structure:

```
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

---

For more information about the overall project, see the main [README](../README.md).
