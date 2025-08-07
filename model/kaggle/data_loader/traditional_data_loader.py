"""
Traditional data loader for hair segmentation dataset.
"""
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from typing import Tuple
from pathlib import Path

from config import TRAINING_CONFIG, DATA_CONFIG
from data_loader.base_data_loader import BaseDataLoader
from data_loader.traditional_dataset import TraditionalDataset

class TraditionalDataLoader(BaseDataLoader):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.use_augmentation = kwargs.get('use_augmentation', DATA_CONFIG.get('use_augmentation', False))
        self.train_images, self.train_masks = None, None
        self.val_images, self.val_masks = None, None

    def load_data(self) -> Tuple[np.ndarray, np.ndarray]:
        print("Loading dataset into memory...")
        image_paths, mask_paths = self.get_file_paths()
        
        images = [self._load_image(Path(p)) for p in tqdm(image_paths, desc="Loading Images")]
        masks = [self._load_mask(Path(p)) for p in tqdm(mask_paths, desc="Loading Masks")]
        
        # Filter out None values from failed loads
        images, masks = zip(*[(i, m) for i, m in zip(images, masks) if i is not None and m is not None])
        
        return np.array(images), np.array(masks)

    def get_datasets(self, validation_split: float = TRAINING_CONFIG["validation_split"], random_seed: int = TRAINING_CONFIG["random_seed"]) -> Tuple[TraditionalDataset, TraditionalDataset]:
        images, masks = self.load_data()
        
        train_images, val_images, train_masks, val_masks = train_test_split(
            images, masks, test_size=validation_split, random_state=random_seed, shuffle=True
        )
        
        # Store for get_data_info method
        self.train_images, self.train_masks = train_images, train_masks
        self.val_images, self.val_masks = val_images, val_masks
        
        print(f"Train set: {len(train_images)} samples, Validation set: {len(val_images)} samples")
        
        train_dataset = TraditionalDataset(train_images, train_masks, use_augmentation=self.use_augmentation)
        val_dataset = TraditionalDataset(val_images, val_masks, use_augmentation=False)
        
        return train_dataset, val_dataset
    
    def get_data_info(self) -> dict:
        """Get information about the dataset."""
        info = {
            "image_size": self.image_size,
            "normalization_factor": self.normalization_factor,
            "train_samples": len(self.train_images) if self.train_images is not None else 0,
            "val_samples": len(self.val_images) if self.val_images is not None else 0,
            "total_samples": len(self.train_images) + len(self.val_images) if self.train_images is not None and self.val_images is not None else 0,
            "augmentation_enabled": self.use_augmentation
        }
        return info

def create_traditional_data_loader(**kwargs) -> TraditionalDataLoader:
    return TraditionalDataLoader(**kwargs)