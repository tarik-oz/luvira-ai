"""
Lazy data loader for hair segmentation dataset.
Only stores file paths and loads data on-demand.
"""

import numpy as np
from pathlib import Path
from typing import Tuple
from sklearn.model_selection import train_test_split
import logging

try:
    from ..config import TRAINING_CONFIG
    from .base_data_loader import BaseDataLoader
    from .lazy_dataset import LazyDataset
except ImportError:
    from config import TRAINING_CONFIG
    from .base_data_loader import BaseDataLoader
    from .lazy_dataset import LazyDataset

logger = logging.getLogger(__name__)

class LazyDataLoader(BaseDataLoader):
    """
    Lazy data loader that only stores file paths and loads data on-demand.
    
    Inherits from BaseDataLoader and implements concrete load_data and split_data methods.
    """
    
    def __init__(self, **kwargs):
        """Initialize lazy data loader."""
        super().__init__(**kwargs)
        
        # Path storage for lazy loading
        self.train_image_paths = []
        self.train_mask_paths = []
        self.val_image_paths = []
        self.val_mask_paths = []
        
    def load_data(self) -> Tuple[list, list]:
        """
        Load file paths only (no actual data loading).
        
        Returns:
            Tuple of (image_paths, mask_paths)
        """
        logger.info("Loading file paths for lazy loading...")
        
        # Get file paths
        image_paths, mask_paths = self.get_file_paths()
        
        logger.info(f"Loaded {len(image_paths)} file paths for lazy loading")
        
        return image_paths, mask_paths
    
    def split_data(self, 
                   validation_split: float = TRAINING_CONFIG["validation_split"],
                   random_seed: int = TRAINING_CONFIG["random_seed"]) -> Tuple[list, list, list, list]:
        """
        Split file paths into training and validation sets.
        
        Args:
            validation_split: Fraction of data to use for validation
            random_seed: Random seed for reproducibility
            
        Returns:
            Tuple of (train_image_paths, train_mask_paths, val_image_paths, val_mask_paths)
        """
        logger.info(f"Splitting file paths with validation_split={validation_split}")
        
        # Get file paths
        image_paths, mask_paths = self.get_file_paths()
        
        # Split paths into train and validation
        train_img_paths, val_img_paths, train_mask_paths, val_mask_paths = train_test_split(
            image_paths, 
            mask_paths, 
            test_size=validation_split,
            random_state=random_seed,
            shuffle=True
        )
        
        # Store paths
        self.train_image_paths = train_img_paths
        self.train_mask_paths = train_mask_paths
        self.val_image_paths = val_img_paths
        self.val_mask_paths = val_mask_paths
        
        logger.info(f"Train set: {len(train_img_paths)} samples")
        logger.info(f"Validation set: {len(val_img_paths)} samples")
        
        return train_img_paths, train_mask_paths, val_img_paths, val_mask_paths
    
    def create_datasets(self, validation_split: float = TRAINING_CONFIG["validation_split"],
                       random_seed: int = TRAINING_CONFIG["random_seed"]) -> Tuple[LazyDataset, LazyDataset]:
        """
        Create train and validation datasets with lazy loading.
        
        Args:
            validation_split: Fraction of data to use for validation
            random_seed: Random seed for reproducibility
            
        Returns:
            Tuple of (train_dataset, val_dataset)
        """
        # Split paths
        train_img_paths, train_mask_paths, val_img_paths, val_mask_paths = self.split_data(
            validation_split, random_seed
        )
        
        # Create datasets
        train_dataset = LazyDataset(
            train_img_paths, 
            train_mask_paths,
            self.image_size, 
            self.normalization_factor
        )
        
        val_dataset = LazyDataset(
            val_img_paths, 
            val_mask_paths,
            self.image_size, 
            self.normalization_factor
        )
        
        logger.info(f"Created lazy datasets - Train: {len(train_dataset)}, Val: {len(val_dataset)}")
        
        return train_dataset, val_dataset
    
    def get_data_info(self) -> dict:
        """
        Get information about the dataset.
        
        Returns:
            Dictionary containing dataset information
        """
        info = super().get_data_info()
        
        # Lazy loading specific information
        if self.train_image_paths and self.val_image_paths:
            info.update({
                "lazy_loading": True,
                "train_samples": len(self.train_image_paths),
                "val_samples": len(self.val_image_paths),
                "total_samples": len(self.train_image_paths) + len(self.val_image_paths),
                "memory_efficient": True,
                "data_loaded": False  # No data in memory
            })
        else:
            info.update({
                "lazy_loading": True,
                "train_samples": 0,
                "val_samples": 0,
                "total_samples": 0,
                "memory_efficient": True,
                "data_loaded": False
            })
        
        return info


def create_lazy_data_loader(**kwargs) -> LazyDataLoader:
    """
    Create a lazy data loader instance with the given configuration.
    
    Args:
        **kwargs: Configuration parameters to override defaults
        
    Returns:
        LazyDataLoader instance
    """
    return LazyDataLoader(**kwargs) 