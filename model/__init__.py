"""
Hair Segmentation U-Net Package with PyTorch

A comprehensive implementation of U-Net for hair segmentation using PyTorch with advanced features.
Features include:
- Timestamped model archiving with config.json and training logs
- Enhanced dataset information and preprocessing
- Comprehensive training pipeline with early stopping and advanced losses
- Flexible inference system supporting different model configurations
- FastAPI REST API for real-time hair segmentation (demo version)
- Singleton model manager for efficient memory usage
- Hair color change functionality (demo version)
- Attention U-Net model with GroupNorm and Dropout
- Windows GPU support and optimized performance
- Support for both grayscale and binary masks
"""

__version__ = "1.1.0"
__author__ = "Tarik"
__description__ = "Comprehensive U-Net implementation for hair segmentation using PyTorch with FastAPI REST API and hair color change features"

from .models.unet_model import UNetModel, create_unet_model
from .models.attention_unet_model import AttentionUNetModel, create_attention_unet_model
from .data.data_loader import HairSegmentationDataLoader, create_data_loader
from .training.trainer import HairSegmentationTrainer, create_trainer
from .inference.predictor import HairSegmentationPredictor, create_predictor

__all__ = [
    "UNetModel",
    "create_unet_model", 
    "AttentionUNetModel",
    "create_attention_unet_model",
    "HairSegmentationDataLoader",
    "create_data_loader",
    "HairSegmentationTrainer", 
    "create_trainer",
    "HairSegmentationPredictor",
    "create_predictor",
] 