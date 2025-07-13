"""
Factory for creating data loaders based on configuration.
Automatically selects between traditional and lazy loading.
"""

import logging
from typing import Union

try:
    from ..config import DATA_CONFIG
    from .traditional_data_loader import TraditionalDataLoader, create_traditional_data_loader
    from .lazy_data_loader import LazyDataLoader, create_lazy_data_loader
    from .base_data_loader import BaseDataLoader
except ImportError:
    from config import DATA_CONFIG
    from .traditional_data_loader import TraditionalDataLoader, create_traditional_data_loader
    from .lazy_data_loader import LazyDataLoader, create_lazy_data_loader
    from .base_data_loader import BaseDataLoader

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
            
        Returns:
            TraditionalDataLoader or LazyDataLoader instance
        """
        # Check if lazy loading is enabled
        lazy_loading = kwargs.pop('lazy_loading', DATA_CONFIG.get('lazy_loading', False))
        
        if lazy_loading:
            logger.info("Creating LazyDataLoader (memory efficient)")
            return create_lazy_data_loader(**kwargs)
        else:
            logger.info("Creating TraditionalDataLoader (loads all data into memory)")
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
            ]
        }

# Convenience function
def create_auto_data_loader(**kwargs) -> Union[TraditionalDataLoader, LazyDataLoader]:
    """
    Auto convenience function to create a data loader based on config.
    
    Args:
        **kwargs: Configuration parameters
        
    Returns:
        Appropriate data loader instance
    """
    return FactoryDataLoader.create_data_loader(**kwargs) 