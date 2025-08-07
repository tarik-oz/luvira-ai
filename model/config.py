"""
Configuration file for Hair Segmentation U-Net project.
Contains all configurable parameters and paths.
"""

import os
from pathlib import Path

# Project root directory
PROJECT_ROOT = Path(__file__).parent

# Dataset path
DATASET_DIR = "small"

# Data paths
DATA_DIR = PROJECT_ROOT / "datasets"
IMAGES_DIR = DATA_DIR / DATASET_DIR / "images"
MASKS_DIR = DATA_DIR / DATASET_DIR / "masks"
PROCESSED_DATA_DIR = DATA_DIR / DATASET_DIR / "processed"

# Trained models paths
TRAINED_MODELS_DIR = PROJECT_ROOT / "trained_models"

# Default model path for inference
DEFAULT_MODEL_PATH = TRAINED_MODELS_DIR / "2025-07-14_02-57-22_acc0.9708/best_model.pth"

# Test paths
TEST_IMAGES_DIR = PROJECT_ROOT / "test_images"
TEST_RESULTS_DIR = PROJECT_ROOT / "test_results"

# Create directories if they don't exist
for directory in [DATA_DIR, IMAGES_DIR, MASKS_DIR, PROCESSED_DATA_DIR, 
                  TRAINED_MODELS_DIR, TEST_IMAGES_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Model configuration
MODEL_CONFIG = {
    "input_shape": (3, 256, 256),  # (channels, height, width)
    "num_filters": [64, 128, 256, 512],
    "bridge_filters": 256,
    "output_channels": 1,
    "activation": "sigmoid", # sigmoid or softmax
    "model_type": "attention_unet"  # unet or attention_unet
}

# Training configuration
TRAINING_CONFIG = {
    "batch_size": 8,
    "epochs": 18,
    "learning_rate": 2e-4,
    "validation_split": 0.2,
    "random_seed": 42,
    "loss_function": "total",  # bce, focal, combo or total
    "bce_weight": 0.3,  # For combo/total loss
    "dice_weight": 0.3, # For combo/total loss
    "boundary_weight": 0.4, # For total loss
    "optimizer": "adam", # adam, adamw or sgd
    "device": "auto",  # auto, cpu, or cuda
    
    # Checkpoint configuration
    "resume_training": False,  # True/False - Resume from checkpoint
    "checkpoint_path": None,  # Model folder name (will be resolved to TRAINED_MODELS_DIR/folder_name)
}

# Data preprocessing configuration
DATA_CONFIG = {
    "image_size": (256, 256),
    "normalization_factor": 255.0,
    "mask_threshold": 0.5,
    "lazy_loading": False,  # True or False (Enable lazy loading for memory efficiency)
    "use_augmentation": True,  # True or False (Enable data augmentation)
    "num_workers": 0  # 0 for Windows, 2-8 for Linux/Mac
}

# Callbacks configuration
CALLBACKS_CONFIG = {
    "checkpoint_monitor": "val_dice", # val_dice or val_loss
    "reduce_lr_monitor": "val_loss", # val_dice or val_loss
    "reduce_lr_patience": 3,
    "reduce_lr_factor": 0.1,
    "reduce_lr_min_lr": 1e-6,
    "early_stopping_monitor": "val_dice", # val_dice or val_loss
    "early_stopping_patience": 5
}

# File patterns
FILE_PATTERNS = {
    "images": ["*.jpg", "*.jpeg", "*.png"],
    "masks": ["*.webp", "*.png", "*.jpg"],
    "processed_images": "train_images.npy",
    "processed_masks": "train_masks.npy",
    "validation_images": "val_images.npy",
    "validation_masks": "val_masks.npy"
}
