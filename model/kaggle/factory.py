"""
Kaggle-specific factory for creating data loaders.
Uses Kaggle configuration and paths.
Enhanced with data augmentation capabilities.
"""

import logging
from typing import Union
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

try:
    from .config import DATA_CONFIG
    from .data_loader import KaggleDataLoader, create_kaggle_data_loader
    from ..data_loader.base_data_loader import BaseDataLoader
except ImportError:
    # Fallback for direct execution
    from config import DATA_CONFIG
    from data_loader import KaggleDataLoader, create_kaggle_data_loader
    from model.data_loader.base_data_loader import BaseDataLoader

logger = logging.getLogger(__name__)

class KaggleFactoryDataLoader:
    """
    Kaggle-specific factory class for creating data loaders.
    
    Always uses lazy loading for memory efficiency in Kaggle environment.
    Includes built-in data augmentation for improved model training.
    """
    
    @staticmethod
    def create_data_loader(**kwargs) -> KaggleDataLoader:
        """
        Create a Kaggle data loader with augmentation support.
        
        Args:
            **kwargs: Configuration parameters to override defaults
                - apply_augmentations: Whether to apply data augmentation (default: True)
            
        Returns:
            KaggleDataLoader instance (always lazy loading)
        """
        # Force lazy loading for Kaggle environment
        lazy_loading = kwargs.pop('lazy_loading', True)
        
        if not lazy_loading:
            logger.warning("Forcing lazy loading for Kaggle environment (memory efficiency)")
        
        # Default to True for augmentations
        enable_augmentations = kwargs.get('apply_augmentations', True)
        if enable_augmentations:
            logger.info("Data augmentation enabled for training")
            logger.info("Applied augmentations: HorizontalFlip, RandomRotation, ColorJitter, RandomBrightnessContrast, ElasticTransform")
        else:
            logger.info("Data augmentation disabled")
        
        logger.info("Creating KaggleDataLoader (memory efficient)")
        return create_kaggle_data_loader(**kwargs)
    
    @staticmethod
    def get_loader_info() -> dict:
        """
        Get information about Kaggle data loader.
        
        Returns:
            Dictionary with loader information
        """
        return {
            "platform": "Kaggle",
            "available_loaders": ["kaggle_lazy"],
            "default_loader": "kaggle_lazy",
            "lazy_loading_enabled": True,
            "memory_efficient": True,
            "kaggle_optimized": True,
            "data_augmentation": True,
            "augmentations": [
                "HorizontalFlip",
                "RandomRotation (±10°)",
                "ColorJitter", 
                "RandomBrightnessContrast",
                "ElasticTransform (mild)"
            ],
            "features": [
                "Memory efficient lazy loading",
                "Kaggle path optimization",
                "Large dataset support",
                "GPU memory optimization",
                "Automatic dataset verification",
                "Real-time data augmentation",
                "Hair-specific augmentation optimization"
            ]
        }

# Convenience function for Kaggle
def create_kaggle_auto_data_loader(**kwargs) -> KaggleDataLoader:
    """
    Auto convenience function to create a Kaggle data loader with augmentation.
    
    Args:
        **kwargs: Configuration parameters
        
    Returns:
        KaggleDataLoader instance
    """
    return KaggleFactoryDataLoader.create_data_loader(**kwargs)

# Override the regular factory function for Kaggle
def create_auto_data_loader(**kwargs) -> KaggleDataLoader:
    """
    Create a data loader optimized for Kaggle environment with augmentation.
    
    Args:
        **kwargs: Configuration parameters
        
    Returns:
        KaggleDataLoader instance
    """
    return create_kaggle_auto_data_loader(**kwargs) 