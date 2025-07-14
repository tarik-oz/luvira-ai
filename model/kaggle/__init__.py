"""
Kaggle module for hair segmentation training.
Contains all Kaggle-specific configurations and training components.
"""

__version__ = "1.0.0"
__author__ = "Tarik"

# Export main components
from .config import *
from .trainer import create_kaggle_trainer
from .data_loader import KaggleDataLoader, create_kaggle_data_loader
from .factory import create_auto_data_loader 