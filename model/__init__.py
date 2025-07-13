"""
Hair Segmentation U-Net Package with PyTorch

Flexible and modular U-Net implementation for hair segmentation tasks using PyTorch.
Includes tools for dataset management, model training, evaluation, and inference.
Highlights:
- Timestamped model archiving with config and training logs
- Enhanced dataset preprocessing and information utilities
- Training pipeline with early stopping and advanced loss functions
- Inference system supporting multiple model configurations
- Singleton model manager for efficient resource usage
- Attention U-Net variant with GroupNorm and Dropout
- Windows GPU support and optimized performance
- Supports both grayscale and binary masks
"""

try:
    from .. import __version__, __author__, __description__
except ImportError:
    # Fallback for when running as top-level module
    __version__ = "2.0.0"
    __author__ = "Tarik"
    __description__ = "Flexible and modular U-Net implementation for hair segmentation tasks using PyTorch."

from .models.unet_model import UNetModel, create_unet_model
from .models.attention_unet_model import AttentionUNetModel, create_attention_unet_model
from .data_loader.lazy_dataset import LazyDataset
from .data_loader.base_data_loader import BaseDataLoader
from .data_loader.traditional_data_loader import TraditionalDataLoader, create_traditional_data_loader
from .data_loader.lazy_data_loader import LazyDataLoader, create_lazy_data_loader
from .data_loader.factory_data_loader import FactoryDataLoader, create_auto_data_loader
from .training.trainer import HairSegmentationTrainer, create_trainer
from .inference.predictor import HairSegmentationPredictor, create_predictor

__all__ = [
    "UNetModel",
    "create_unet_model", 
    "AttentionUNetModel",
    "create_attention_unet_model",
    "LazyDataset",
    "BaseDataLoader",
    "TraditionalDataLoader",
    "create_traditional_data_loader",
    "LazyDataLoader",
    "create_lazy_data_loader",
    "FactoryDataLoader",
    "create_auto_data_loader",
    "HairSegmentationTrainer", 
    "create_trainer",
    "HairSegmentationPredictor",
    "create_predictor",
] 