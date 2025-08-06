"""
Kaggle module for hair segmentation training.
Contains all Kaggle-specific configurations and training components.
"""

__version__ = "1.0.0"
__author__ = "Tarik"

# Export main components (only existing modules)
from .config import *

__all__ = [
    "TRAINING_CONFIG",
    "MODEL_CONFIG", 
    "DATA_CONFIG",
    "CALLBACKS_CONFIG",
    "DATASET_NAME",
    "IMAGES_DIR",
    "MASKS_DIR",
    "TRAINED_MODELS_DIR"
] 