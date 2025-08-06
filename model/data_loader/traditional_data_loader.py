"""
Traditional data loader for hair segmentation dataset.
Loads all data into memory at once.
"""

import numpy as np
import torch
import cv2
from torch.utils.data import Dataset
from pathlib import Path
from typing import Tuple, List
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import logging

from model.config import TRAINING_CONFIG, DATA_CONFIG
from model.utils.data_timestamp import get_latest_timestamp, save_timestamps, load_timestamps, needs_processing
from model.utils.augmentation import Augmentation
from model.data_loader.base_data_loader import BaseDataLoader
from model.data_loader.traditional_dataset import TraditionalDataset

logger = logging.getLogger(__name__)

class TraditionalDataLoader(BaseDataLoader):
    """
    Traditional data loader that loads all data into memory at once.
    
    Inherits from BaseDataLoader and implements concrete load_data and split_data methods.
    """
    
    def __init__(self, **kwargs):
        """Initialize traditional data loader."""
        # Pop augmentation setting before calling super
        self.use_augmentation = kwargs.pop('use_augmentation', DATA_CONFIG.get('use_augmentation', False))

        super().__init__(**kwargs)
        
        # Data storage for traditional loading
        self.images = []
        self.masks = []
        self.train_images = None
        self.train_masks = None
        self.val_images = None
        self.val_masks = None
        
        if self.use_augmentation:
            logger.info("Data augmentation enabled for traditional loading")
        
    def load_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load all images and masks from the dataset into memory.
        
        Returns:
            Tuple of (images, masks) as numpy arrays
        """
        logger.info("Loading dataset into memory...")
        
        # Get file paths
        image_paths, mask_paths = self.get_file_paths()
        
        # Load images and masks
        images = []
        masks = []
        
        for img_path, mask_path in tqdm(zip(image_paths, mask_paths), 
                                       total=min(len(image_paths), len(mask_paths)), 
                                       desc="Loading data"):
            image = self._load_image(Path(img_path))
            mask = self._load_mask(Path(mask_path))
            
            if image is not None and mask is not None:
                images.append(image)
                masks.append(mask)
        
        # Convert to numpy arrays
        self.images = np.array(images)
        self.masks = np.array(masks)
        
        logger.info(f"Loaded {len(self.images)} images and masks")
        logger.info(f"Images shape: {self.images.shape}")
        logger.info(f"Masks shape: {self.masks.shape}")
        
        return self.images, self.masks
    
    def split_data(self, 
                   validation_split: float = TRAINING_CONFIG["validation_split"],
                   random_seed: int = TRAINING_CONFIG["random_seed"]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Split data into training and validation sets.
        
        Args:
            validation_split: Fraction of data to use for validation
            random_seed: Random seed for reproducibility
            
        Returns:
            Tuple of (train_images, train_masks, val_images, val_masks)
        """
        if self.images is None or self.masks is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        logger.info(f"Splitting data with validation_split={validation_split}")
        
        self.train_images, self.val_images, self.train_masks, self.val_masks = train_test_split(
            self.images,
            self.masks, 
            test_size=validation_split,
            random_state=random_seed,
            shuffle=True
        )
        
        logger.info(f"Train set: {len(self.train_images)} samples")
        logger.info(f"Validation set: {len(self.val_images)} samples")
        
        return self.train_images, self.train_masks, self.val_images, self.val_masks
    
    def save_processed_data(self) -> None:
        """
        Save processed data to numpy files.
        """
        if self.train_images is None or self.val_images is None:
            raise ValueError("Data not split. Call split_data() first.")
        
        logger.info("Saving processed data...")
        
        # Save training data
        train_images_path = self.processed_dir / "train_images.npy"
        train_masks_path = self.processed_dir / "train_masks.npy"
        
        np.save(train_images_path, self.train_images)
        np.save(train_masks_path, self.train_masks)
        
        # Save validation data
        val_images_path = self.processed_dir / "val_images.npy"
        val_masks_path = self.processed_dir / "val_masks.npy"
        
        np.save(val_images_path, self.val_images)
        np.save(val_masks_path, self.val_masks)

        # Save timestamps
        image_ts = get_latest_timestamp(self.images_dir)
        mask_ts = get_latest_timestamp(self.masks_dir)
        save_timestamps(self.processed_dir, image_ts, mask_ts)
        
        logger.info(f"Processed data saved to {self.processed_dir}")
    
    def load_processed_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Load processed data from disk with fallback to reprocessing.
        
        Returns:
            Tuple of (train_images, train_masks, val_images, val_masks)
        """
        # Try to load existing
        if not needs_processing(self.images_dir, self.masks_dir, self.processed_dir):
            try:
                return self._load_existing_data()
            except Exception as e:
                logger.warning(f"Failed to load existing data: {e}")
        
        # Reprocess
        logger.info("Processing data from scratch...")
        try:
            return self._reprocess_data()
        except Exception as e:
            logger.error(f"Failed to process data: {e}")
            raise RuntimeError(f"Data processing failed: {e}")

    def _load_existing_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Load existing processed data from numpy files.
        
        Returns:
            Tuple of (train_images, train_masks, val_images, val_masks)
        """
        logger.info("Loading up-to-date processed data...")
        
        # Prepare paths
        train_images_path = self.processed_dir / "train_images.npy"
        train_masks_path = self.processed_dir / "train_masks.npy"
        val_images_path = self.processed_dir / "val_images.npy"
        val_masks_path = self.processed_dir / "val_masks.npy"
        
        # Load data
        train_images = np.load(train_images_path)
        train_masks = np.load(train_masks_path)
        val_images = np.load(val_images_path)
        val_masks = np.load(val_masks_path)
        
        # Self assignment
        self.train_images = train_images
        self.train_masks = train_masks
        self.val_images = val_images
        self.val_masks = val_masks
        
        logger.info("Processed data loaded successfully")
        logger.info(f"Train set: {len(train_images)} samples")
        logger.info(f"Validation set: {len(val_images)} samples")
        
        return train_images, train_masks, val_images, val_masks

    def _reprocess_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Reprocess data from scratch.
        
        Returns:
            Tuple of (train_images, train_masks, val_images, val_masks)
        """
        logger.info("Reprocessing data from scratch...")
        
        self.load_data()
        data = self.split_data()
        
        self.save_processed_data()
        
        logger.info("Data reprocessing completed successfully")
        return data
    
    def get_datasets(self, validation_split: float = TRAINING_CONFIG["validation_split"],
                    random_seed: int = TRAINING_CONFIG["random_seed"]) -> Tuple[Dataset, Dataset]:
        """
        Get PyTorch datasets for training and validation.
        
        Args:
            validation_split: Fraction of data to use for validation
            random_seed: Random seed for reproducibility
            
        Returns:
            Tuple of (train_dataset, val_dataset)
        """
        # Load processed data
        train_images, train_masks, val_images, val_masks = self.load_processed_data()
        
        # Create custom datasets with augmentation support
        train_dataset = TraditionalDataset(
            train_images, 
            train_masks,
            use_augmentation=self.use_augmentation,
            is_training=True
        )
        
        val_dataset = TraditionalDataset(
            val_images, 
            val_masks,
            use_augmentation=False,  # No augmentation for validation
            is_training=False
        )
        
        return train_dataset, val_dataset
    
    def get_data_info(self) -> dict:
        """
        Get information about the dataset.
        
        Returns:
            Dictionary containing dataset information
        """
        info = super().get_data_info()
        
        # Add sample counts and augmentation info
        if self.train_images is not None and self.val_images is not None:
            info.update({
                "train_samples": len(self.train_images),
                "val_samples": len(self.val_images),
                "total_samples": len(self.train_images) + len(self.val_images),
                "augmentation_enabled": self.use_augmentation,
                "augmentations": [
                    "HorizontalFlip",
                    "RandomRotation",
                    "ColorJitter",
                    "RandomBrightnessContrast"
                ] if self.use_augmentation else []
            })
        else:
            info.update({
                "train_samples": 0,
                "val_samples": 0,
                "total_samples": 0,
                "augmentation_enabled": self.use_augmentation
            })
        
        return info


def create_traditional_data_loader(**kwargs) -> TraditionalDataLoader:
    """
    Create a traditional data loader instance with the given configuration.
    
    Args:
        **kwargs: Configuration parameters to override defaults
        
    Returns:
        TraditionalDataLoader instance
    """
    return TraditionalDataLoader(**kwargs) 