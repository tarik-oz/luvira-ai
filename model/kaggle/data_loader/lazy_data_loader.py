"""
Lazy data loader for hair segmentation dataset.
Only stores file paths and loads data on-demand.
"""

from typing import Tuple
from sklearn.model_selection import train_test_split

from config import TRAINING_CONFIG, DATA_CONFIG
from data_loader.base_data_loader import BaseDataLoader
from data_loader.lazy_dataset import LazyDataset

class LazyDataLoader(BaseDataLoader):
    """
    Lazy data loader that only stores file paths and loads data on-demand.
    """
    
    def __init__(self, **kwargs):
        """Initialize lazy data loader."""
        super().__init__(**kwargs)
        self.use_augmentation = kwargs.get('use_augmentation', DATA_CONFIG.get('use_augmentation', False))
        self.train_image_paths, self.train_mask_paths = [], []
        self.val_image_paths, self.val_mask_paths = [], []

    def get_datasets(self, validation_split: float = TRAINING_CONFIG["validation_split"], random_seed: int = TRAINING_CONFIG["random_seed"]) -> Tuple[LazyDataset, LazyDataset]:
        image_paths, mask_paths = self.get_file_paths()
        
        train_img_paths, val_img_paths, train_mask_paths, val_mask_paths = train_test_split(
            image_paths, mask_paths, test_size=validation_split, random_state=random_seed, shuffle=True
        )
        
        # Store paths for get_data_info method
        self.train_image_paths, self.train_mask_paths = train_img_paths, train_mask_paths
        self.val_image_paths, self.val_mask_paths = val_img_paths, val_mask_paths
        
        print(f"Train set: {len(train_img_paths)} samples, Validation set: {len(val_img_paths)} samples")

        train_dataset = LazyDataset(train_img_paths, train_mask_paths, self.image_size, self.normalization_factor, use_augmentation=self.use_augmentation)
        val_dataset = LazyDataset(val_img_paths, val_mask_paths, self.image_size, self.normalization_factor, use_augmentation=False)
        
        return train_dataset, val_dataset
    
    def get_data_info(self) -> dict:
        """Get information about the dataset."""
        info = {
            "image_size": self.image_size,
            "normalization_factor": self.normalization_factor,
            "train_samples": len(self.train_image_paths) if hasattr(self, 'train_image_paths') else 0,
            "val_samples": len(self.val_image_paths) if hasattr(self, 'val_image_paths') else 0,
            "total_samples": len(self.train_image_paths) + len(self.val_image_paths) if hasattr(self, 'train_image_paths') else 0,
            "augmentation_enabled": self.use_augmentation,
        }
        return info

def create_lazy_data_loader(**kwargs) -> LazyDataLoader:
    return LazyDataLoader(**kwargs)