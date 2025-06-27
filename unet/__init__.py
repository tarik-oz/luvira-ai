"""
Hair Segmentation U-Net Package with PyTorch

A clean and modular implementation of U-Net for hair segmentation using PyTorch.
Features include:
- Timestamped model archiving with config.json and training logs
- Enhanced dataset information and preprocessing
- Comprehensive training pipeline with early stopping
- Flexible inference system supporting different model configurations
- Windows GPU support and optimized performance
"""

__version__ = "1.0.0"
__author__ = "Tarik"
__description__ = "Clean and modular U-Net implementation for hair segmentation using PyTorch with advanced archiving system"

from .models.unet_model import UNetModel, create_unet_model
from .data.data_loader import HairSegmentationDataLoader, create_data_loader
from .training.trainer import HairSegmentationTrainer, create_trainer
from .inference.predictor import HairSegmentationPredictor, create_predictor

__all__ = [
    "UNetModel",
    "create_unet_model", 
    "HairSegmentationDataLoader",
    "create_data_loader",
    "HairSegmentationTrainer", 
    "create_trainer",
    "HairSegmentationPredictor",
    "create_predictor"
] 