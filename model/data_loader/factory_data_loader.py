"""
Factory for creating data loaders based on configuration.
Automatically selects between traditional and lazy loading with augmentation options.
"""

import logging
from typing import Union

from model.config import DATA_CONFIG
from model.data_loader.traditional_data_loader import TraditionalDataLoader, create_traditional_data_loader
from model.data_loader.lazy_data_loader import LazyDataLoader, create_lazy_data_loader

logger = logging.getLogger(__name__)

class FactoryDataLoader:
    """
    Factory class for creating data loaders.
    
    Automatically selects the appropriate loader based on configuration.
    """
    
    @staticmethod
    def create_data_loader(**kwargs) -> Union[TraditionalDataLoader, LazyDataLoader]:
        """
        Create a data loader based on configuration.
        
        Args:
            **kwargs: Configuration parameters to override defaults
                - lazy_loading: Whether to use lazy loading
                - use_augmentation: Whether to use data augmentation
            
        Returns:
            Appropriate data loader instance
        """
        # Check if lazy loading is enabled
        lazy_loading = kwargs.pop('lazy_loading', DATA_CONFIG.get('lazy_loading', False))
        
        # Check if augmentation is enabled
        use_augmentation = kwargs.pop('use_augmentation', DATA_CONFIG.get('use_augmentation', False))
        
        # Add augmentation setting back to kwargs for the data loader
        kwargs['use_augmentation'] = use_augmentation
        
        if lazy_loading:
            logger.info("Creating LazyDataLoader (memory efficient)")
            if use_augmentation:
                logger.info("Data augmentation enabled")
            return create_lazy_data_loader(**kwargs)
        else:
            logger.info("Creating TraditionalDataLoader (loads all data into memory)")
            if use_augmentation:
                logger.info("Data augmentation enabled")
            return create_traditional_data_loader(**kwargs)
    
    @staticmethod
    def get_loader_info() -> dict:
        """
        Get information about available loaders.
        
        Returns:
            Dictionary with loader information
        """
        return {
            "available_loaders": ["traditional", "lazy"],
            "default_loader": "lazy" if DATA_CONFIG.get('lazy_loading', False) else "traditional",
            "lazy_loading_enabled": DATA_CONFIG.get('lazy_loading', False),
            "augmentation_enabled": DATA_CONFIG.get('use_augmentation', False),
            "traditional_features": [
                "Loads all data into memory",
                "Faster training (no disk I/O during training)",
                "Higher memory usage",
                "Suitable for small datasets"
            ],
            "lazy_features": [
                "Loads data on-demand",
                "Memory efficient",
                "Slower training (disk I/O during training)",
                "Suitable for large datasets"
            ],
            "augmentation_features": [
                "Real-time data augmentation",
                "Enhanced training with data variety",
                "Includes: HorizontalFlip, RandomRotation, ColorJitter, RandomBrightnessContrast"
            ]
        }


def create_auto_data_loader(**kwargs) -> Union[TraditionalDataLoader, LazyDataLoader]:
    """
    Create a data loader automatically based on configuration.
    
    Args:
        **kwargs: Configuration parameters to override defaults
        
    Returns:
        Appropriate data loader instance
    """
    return FactoryDataLoader.create_data_loader(**kwargs) 