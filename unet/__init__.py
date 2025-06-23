"""
Hair Segmentation U-Net Package

A clean and modular implementation of U-Net for hair segmentation.
"""

__version__ = "2.0.0"
__author__ = "Tarik"
__description__ = "Clean and modular U-Net implementation for hair segmentation"

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