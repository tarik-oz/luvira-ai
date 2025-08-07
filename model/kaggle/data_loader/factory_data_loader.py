"""
Factory for creating data loaders based on configuration.
"""

from typing import Union

from config import DATA_CONFIG
from data_loader.traditional_data_loader import TraditionalDataLoader, create_traditional_data_loader
from data_loader.lazy_data_loader import LazyDataLoader, create_lazy_data_loader

def create_auto_data_loader(lazy_loading: bool, **kwargs) -> Union[TraditionalDataLoader, LazyDataLoader]:
    """
    Create a data loader automatically based on configuration.
    """
    print(f"Creating {'Lazy' if lazy_loading else 'Traditional'}DataLoader")
    
    loader_kwargs = {
        'use_augmentation': DATA_CONFIG.get('use_augmentation', False),
        **kwargs
    }
    
    if lazy_loading:
        return create_lazy_data_loader(**loader_kwargs)
    return create_traditional_data_loader(**loader_kwargs)
