"""
Kaggle-specific configuration file for Hair Segmentation U-Net project.
Contains all configurable parameters and paths for Kaggle environment.
UPDATED VERSION - All fixes included!
"""

import os
from pathlib import Path

# Kaggle specific paths
KAGGLE_INPUT_DIR = Path("/kaggle/input")
KAGGLE_WORKING_DIR = Path("/kaggle/working")

# Project root directory (in Kaggle working directory)
PROJECT_ROOT = KAGGLE_WORKING_DIR

# Dataset path - your dataset name on Kaggle
DATASET_NAME = "hair-dataset-99"  # Change this to match your Kaggle dataset name

# Data paths for Kaggle
DATA_DIR = KAGGLE_INPUT_DIR / DATASET_NAME
IMAGES_DIR = DATA_DIR / "images"
MASKS_DIR = DATA_DIR / "masks"
PROCESSED_DATA_DIR = KAGGLE_WORKING_DIR / "processed"

# Trained models paths (save to working directory in Kaggle)
TRAINED_MODELS_DIR = KAGGLE_WORKING_DIR / "trained_models"

# Test paths
TEST_IMAGES_DIR = PROJECT_ROOT / "test_images"
TEST_RESULTS_DIR = KAGGLE_WORKING_DIR / "test_results"

# Create directories if they don't exist
for directory in [PROCESSED_DATA_DIR, TRAINED_MODELS_DIR, TEST_RESULTS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Verify dataset exists
if not IMAGES_DIR.exists():
    raise FileNotFoundError(f"Dataset not found at {IMAGES_DIR}. Please ensure your dataset is named '{DATASET_NAME}' in Kaggle.")

if not MASKS_DIR.exists():
    raise FileNotFoundError(f"Masks not found at {MASKS_DIR}. Please ensure your dataset has 'masks' folder.")

print(f"Dataset found: {DATA_DIR}")
print(f"Images directory: {IMAGES_DIR}")
print(f"Masks directory: {MASKS_DIR}")
print(f"Working directory: {KAGGLE_WORKING_DIR}")

# Model configuration - Optimized for 30K dataset
MODEL_CONFIG = {
    "input_shape": (3, 256, 256),  # (channels, height, width)
    "num_filters": [64, 128, 256, 512],
    "bridge_filters": 512,  # Increased for better feature representation
    "output_channels": 1,
    "activation": "sigmoid", # sigmoid or softmax
    "model_type": "attention_unet"  # unet or attention_unet
}

# Training configuration - Optimized for 30K dataset and Kaggle GPU
TRAINING_CONFIG = {
    "batch_size": 32,  # Increased for 30K dataset
    "epochs": 40,      # More epochs for large dataset
    "learning_rate": 1e-4,  # Slightly lower LR for stability
    "validation_split": 0.15,  # 15% validation for large dataset
    "random_seed": 42,
    "model_type": "attention_unet",  # unet or attention_unet
    "loss_function": "total",  # bce, focal, combo or total
    "bce_weight": 0.2,  # Adjusted weights for 30K dataset
    "dice_weight": 0.4, # Higher dice weight for better segmentation
    "boundary_weight": 0.4, # Keep boundary emphasis
    "optimizer": "adamw", # AdamW better for large datasets
    "device": "auto"  # auto, cpu, or cuda
}

# Data preprocessing configuration - Optimized for 30K dataset
DATA_CONFIG = {
    "image_size": (256, 256),
    "normalization_factor": 255.0,
    "mask_threshold": 0.5,
    "lazy_loading": False,  # Enable lazy loading for large dataset
    "use_augmentation": True,  # Enable data augmentation
    "num_workers": 2  # Kaggle supports 2 workers
}

# Callbacks configuration - optimized for 30K dataset
CALLBACKS_CONFIG = {
    "checkpoint_monitor": "val_dice", # Monitor val_dice for best model
    "early_stopping_monitor": "val_dice", # Early stopping on val_dice
    "early_stopping_patience": 20,  # More patience for large dataset
    "reduce_lr_monitor": "val_loss", # LR reduction on val_loss plateau
    "reduce_lr_patience": 8,   # Faster LR reduction
    "reduce_lr_factor": 0.3,   # More aggressive LR reduction
    "reduce_lr_min_lr": 5e-7   # Lower minimum LR
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

# Log dataset statistics
def log_dataset_stats():
    """Log dataset statistics for monitoring"""
    if IMAGES_DIR.exists() and MASKS_DIR.exists():
        image_files = []
        for pattern in FILE_PATTERNS["images"]:
            image_files.extend(list(IMAGES_DIR.glob(pattern)))
        
        mask_files = []
        for pattern in FILE_PATTERNS["masks"]:
            mask_files.extend(list(MASKS_DIR.glob(pattern)))
        
        print(f"Found {len(image_files)} images and {len(mask_files)} masks")
        
        if len(image_files) != len(mask_files):
            print(f"WARNING: Image and mask counts don't match!")
        
        return len(image_files), len(mask_files)
    return 0, 0

# Log stats at import
log_dataset_stats() 