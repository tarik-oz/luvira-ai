"""
Configuration file for Hair Segmentation U-Net project.
Contains all configurable parameters and paths.
"""

import os
from pathlib import Path

# Project root directory
PROJECT_ROOT = Path(__file__).parent

# Data paths
DATA_DIR = PROJECT_ROOT / "data"
IMAGES_DIR = DATA_DIR / "images"
MASKS_DIR = DATA_DIR / "masks"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# Model paths
MODEL_DIR = PROJECT_ROOT / "models"
TRAINED_MODELS_DIR = MODEL_DIR / "trained_models"

# Test paths
TEST_IMAGES_DIR = PROJECT_ROOT / "test_images"
TEST_RESULTS_DIR = PROJECT_ROOT / "test_results"

# Create directories if they don't exist
for directory in [DATA_DIR, IMAGES_DIR, MASKS_DIR, PROCESSED_DATA_DIR, 
                  MODEL_DIR, TRAINED_MODELS_DIR, TEST_IMAGES_DIR, TEST_RESULTS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Model configuration
MODEL_CONFIG = {
    "input_shape": (3, 256, 256),  # (channels, height, width)
    "num_filters": [64, 128, 256, 512],
    "bridge_filters": 256,
    "output_channels": 1,
    "activation": "sigmoid"
}

# Training configuration
TRAINING_CONFIG = {
    "batch_size": 8,
    "epochs": 50,
    "learning_rate": 1e-4,
    "validation_split": 0.2,
    "random_seed": 42,
    "loss_function": "bce",  # Binary Cross Entropy
    "optimizer": "adam",
    "metrics": ["accuracy"],
    "device": "auto"  # auto, cpu, or cuda
}

# Data preprocessing configuration
DATA_CONFIG = {
    "image_size": (256, 256),
    "normalization_factor": 255.0,
    "mask_threshold": 0.5
}

# Callbacks configuration
CALLBACKS_CONFIG = {
    "checkpoint_monitor": "val_loss",
    "checkpoint_save_best_only": True,
    "reduce_lr_monitor": "val_loss",
    "reduce_lr_patience": 3,
    "reduce_lr_factor": 0.1,
    "reduce_lr_min_lr": 1e-6,
    "early_stopping_monitor": "val_accuracy",
    "early_stopping_patience": 3
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

# API configuration
API_CONFIG = {
    "default_model_path": "models/trained_models/2025-06-28_23-25-10_acc0.9822/best_model.pth",
    "host": "0.0.0.0",
    "port": 8000,
    "reload": True,
    "log_level": "info",
    "cors_origins": ["*"],
    "cors_methods": ["*"],
    "cors_headers": ["*"],
    "max_file_size": 10 * 1024 * 1024,  # 10MB
    "allowed_image_types": ["image/jpeg", "image/png", "image/jpg"]
} 