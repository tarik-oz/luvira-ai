"""
Data loader for hair segmentation dataset.
Handles loading, preprocessing, and splitting of image and mask data.
"""

import cv2
import numpy as np
import glob
from pathlib import Path
from typing import Tuple, List, Optional
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import logging

from config import (
    IMAGES_DIR, MASKS_DIR, PROCESSED_DATA_DIR, 
    DATA_CONFIG, TRAINING_CONFIG, FILE_PATTERNS
)
from utils.data_timestamp import get_latest_timestamp, save_timestamps, load_timestamps, needs_processing

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HairSegmentationDataLoader:
    """
    Data loader for hair segmentation dataset.
    
    Handles loading images and masks, preprocessing, and train/validation splitting.
    """
    
    def __init__(self, 
                 images_dir: Path = IMAGES_DIR,
                 masks_dir: Path = MASKS_DIR,
                 processed_dir: Path = PROCESSED_DATA_DIR,
                 image_size: Tuple[int, int] = DATA_CONFIG["image_size"],
                 normalization_factor: float = DATA_CONFIG["normalization_factor"]):
        """
        Initialize the data loader.
        
        Args:
            images_dir: Directory containing input images
            masks_dir: Directory containing mask images
            processed_dir: Directory to save processed data
            image_size: Target size for images (height, width)
            normalization_factor: Factor to normalize pixel values
        """
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        self.processed_dir = Path(processed_dir)
        self.image_size = image_size
        self.normalization_factor = normalization_factor
        
        # Create processed directory if it doesn't exist
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        
        # Data storage
        self.images = []
        self.masks = []
        self.train_images = None
        self.train_masks = None
        self.val_images = None
        self.val_masks = None
        
    def _load_image(self, image_path: Path) -> np.ndarray:
        """
        Load and preprocess a single image.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Preprocessed image as numpy array
        """
        try:
            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError(f"Could not load image: {image_path}")
                
            # Resize image
            image = cv2.resize(image, self.image_size)
            
            # Normalize pixel values
            image = image / self.normalization_factor
            image = image.astype(np.float32)
            
            return image
            
        except Exception as e:
            logger.error(f"Error loading image {image_path}: {e}")
            return None
    
    def _load_mask(self, mask_path: Path) -> np.ndarray:
        """
        Load and preprocess a single mask.
        
        Args:
            mask_path: Path to the mask file
            
        Returns:
            Preprocessed mask as numpy array (float32, between 0 and 1)
        """
        try:
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if mask is None:
                raise ValueError(f"Could not load mask: {mask_path}")
                
            # Resize mask
            mask = cv2.resize(mask, self.image_size)
            
            # Normalize mask to 0-1 float
            mask = mask.astype(np.float32) / 255.0
            
            return mask
            
        except Exception as e:
            logger.error(f"Error loading mask {mask_path}: {e}")
            return None
    
    def load_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load all images and masks from the dataset.
        
        Returns:
            Tuple of (images, masks) as numpy arrays
        """
        logger.info("Loading dataset...")
        
        # Get file paths for multiple patterns
        image_paths = []
        for pattern in FILE_PATTERNS["images"]:
            pattern_path = str(self.images_dir / pattern)
            image_paths.extend(glob.glob(pattern_path))
        
        mask_paths = []
        for pattern in FILE_PATTERNS["masks"]:
            pattern_path = str(self.masks_dir / pattern)
            mask_paths.extend(glob.glob(pattern_path))
        
        # Sort paths to ensure matching
        image_paths = sorted(image_paths)
        mask_paths = sorted(mask_paths)
        
        if not image_paths or not mask_paths:
            raise ValueError(f"No images or masks found in {self.images_dir} or {self.masks_dir}")
        
        if len(image_paths) != len(mask_paths):
            logger.warning(f"Number of images ({len(image_paths)}) doesn't match number of masks ({len(mask_paths)})")
            logger.warning("This might cause issues. Make sure image and mask files have matching names.")
        
        logger.info(f"Found {len(image_paths)} images and {len(mask_paths)} masks")
        
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
            random_state=random_seed
        )
        
        logger.info(f"Training set: {self.train_images.shape[0]} samples")
        logger.info(f"Validation set: {self.val_images.shape[0]} samples")
        
        return self.train_images, self.train_masks, self.val_images, self.val_masks
    
    def save_processed_data(self) -> None:
        """
        Save processed data to numpy files.
        """
        if (self.train_images is None or self.train_masks is None or 
            self.val_images is None or self.val_masks is None):
            raise ValueError("Data not split. Call split_data() first.")
        
        logger.info("Saving processed data...")
        
        # Save training data
        train_images_path = self.processed_dir / FILE_PATTERNS["processed_images"]
        train_masks_path = self.processed_dir / FILE_PATTERNS["processed_masks"]
        
        np.save(train_images_path, self.train_images)
        np.save(train_masks_path, self.train_masks)
        
        # Save validation data
        val_images_path = self.processed_dir / FILE_PATTERNS["validation_images"]
        val_masks_path = self.processed_dir / FILE_PATTERNS["validation_masks"]
        
        np.save(val_images_path, self.val_images)
        np.save(val_masks_path, self.val_masks)
        
        logger.info(f"Data saved to {self.processed_dir}")
    
    def load_processed_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Load processed data if available and up-to-date, otherwise process raw data.
        Returns:
            Tuple of (train_images, train_masks, val_images, val_masks)
        """
        # Check if processing is needed
        if needs_processing(self.images_dir, self.masks_dir, self.processed_dir):
            logger.info("Detected changes in images or masks. Re-processing data...")
            self.load_data()
            data = self.split_data()
            # Save processed data
            self.save_processed_data()
            # Save new timestamps
            image_ts = get_latest_timestamp(self.images_dir)
            mask_ts = get_latest_timestamp(self.masks_dir)
            save_timestamps(self.processed_dir, image_ts, mask_ts)
            return data
        # Otherwise, load existing processed data
        logger.info("Loading up-to-date processed data...")
        train_images = np.load(self.processed_dir / "train_images.npy")
        train_masks = np.load(self.processed_dir / "train_masks.npy")
        val_images = np.load(self.processed_dir / "val_images.npy")
        val_masks = np.load(self.processed_dir / "val_masks.npy")
        return train_images, train_masks, val_images, val_masks
    
    def get_data_info(self) -> dict:
        """
        Get comprehensive information about the loaded dataset.
        
        Returns:
            Dictionary containing dataset information
        """
        info = {
            "image_size": self.image_size,
            "normalization_factor": self.normalization_factor,
            "images_dir_exists": self.images_dir.exists(),
            "masks_dir_exists": self.masks_dir.exists(),
            "processed_dir_exists": self.processed_dir.exists()
        }
        
        # Raw data information
        if self.images is not None:
            info.update({
                "raw_data_loaded": True,
                "total_samples": len(self.images),
                "raw_data_type": type(self.images).__name__
            })
            
            # Check if data is numpy array or list
            if hasattr(self.images, 'shape'):
                info.update({
                    "raw_images_shape": self.images.shape,
                    "raw_masks_shape": self.masks.shape
                })
            else:
                # Data is still in list format
                info.update({
                    "raw_images_shape": f"list with {len(self.images)} items",
                    "raw_masks_shape": f"list with {len(self.masks)} items"
                })
        else:
            info.update({
                "raw_data_loaded": False,
                "total_samples": 0
            })
        
        # Processed data information
        if self.train_images is not None:
            info.update({
                "processed_data_loaded": True,
                "train_samples": len(self.train_images),
                "val_samples": len(self.val_images),
                "train_images_shape": self.train_images.shape,
                "train_masks_shape": self.train_masks.shape,
                "val_images_shape": self.val_images.shape,
                "val_masks_shape": self.val_masks.shape
            })
        else:
            info.update({
                "processed_data_loaded": False
            })
        
        return info


def create_data_loader(**kwargs) -> HairSegmentationDataLoader:
    """
    Factory function to create a data loader.
    
    Args:
        **kwargs: Arguments to pass to HairSegmentationDataLoader
        
    Returns:
        HairSegmentationDataLoader instance
    """
    return HairSegmentationDataLoader(**kwargs) 