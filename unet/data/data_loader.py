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
            Preprocessed mask as numpy array
        """
        try:
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if mask is None:
                raise ValueError(f"Could not load mask: {mask_path}")
                
            # Resize mask
            mask = cv2.resize(mask, self.image_size)
            
            # Binarize mask (threshold at 0)
            mask[mask > 0] = 1
            mask = mask.astype(np.uint8)
            
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
        
        # Get file paths
        image_pattern = str(self.images_dir / FILE_PATTERNS["images"])
        mask_pattern = str(self.masks_dir / FILE_PATTERNS["masks"])
        
        image_paths = sorted(glob.glob(image_pattern))
        mask_paths = sorted(glob.glob(mask_pattern))
        
        if not image_paths or not mask_paths:
            raise ValueError(f"No images or masks found in {self.images_dir} or {self.masks_dir}")
        
        if len(image_paths) != len(mask_paths):
            raise ValueError(f"Number of images ({len(image_paths)}) doesn't match number of masks ({len(mask_paths)})")
        
        logger.info(f"Found {len(image_paths)} image-mask pairs")
        
        # Load images and masks
        images = []
        masks = []
        
        for img_path, mask_path in tqdm(zip(image_paths, mask_paths), 
                                       total=len(image_paths), 
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
        Load previously processed data from numpy files.
        
        Returns:
            Tuple of (train_images, train_masks, val_images, val_masks)
        """
        logger.info("Loading processed data...")
        
        # Load training data
        train_images_path = self.processed_dir / FILE_PATTERNS["processed_images"]
        train_masks_path = self.processed_dir / FILE_PATTERNS["processed_masks"]
        
        if not train_images_path.exists() or not train_masks_path.exists():
            raise FileNotFoundError("Processed training data not found. Run save_processed_data() first.")
        
        self.train_images = np.load(train_images_path)
        self.train_masks = np.load(train_masks_path)
        
        # Load validation data
        val_images_path = self.processed_dir / FILE_PATTERNS["validation_images"]
        val_masks_path = self.processed_dir / FILE_PATTERNS["validation_masks"]
        
        if not val_images_path.exists() or not val_masks_path.exists():
            raise FileNotFoundError("Processed validation data not found. Run save_processed_data() first.")
        
        self.val_images = np.load(val_images_path)
        self.val_masks = np.load(val_masks_path)
        
        logger.info(f"Loaded training data: {self.train_images.shape}")
        logger.info(f"Loaded validation data: {self.val_images.shape}")
        
        return self.train_images, self.train_masks, self.val_images, self.val_masks
    
    def get_data_info(self) -> dict:
        """
        Get information about the loaded dataset.
        
        Returns:
            Dictionary containing dataset information
        """
        info = {
            "total_samples": len(self.images) if self.images is not None else 0,
            "image_size": self.image_size,
            "normalization_factor": self.normalization_factor
        }
        
        if self.train_images is not None:
            info.update({
                "train_samples": len(self.train_images),
                "val_samples": len(self.val_images),
                "train_images_shape": self.train_images.shape,
                "train_masks_shape": self.train_masks.shape,
                "val_images_shape": self.val_images.shape,
                "val_masks_shape": self.val_masks.shape
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