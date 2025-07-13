"""
Base data loader class for hair segmentation dataset.
Contains common functionality for both traditional and lazy loading approaches.
"""

import cv2
import numpy as np
import glob
from pathlib import Path
from typing import Tuple, List, Optional
from abc import ABC, abstractmethod
import logging

try:
    from ..config import (
        IMAGES_DIR, MASKS_DIR, PROCESSED_DATA_DIR, 
        DATA_CONFIG, TRAINING_CONFIG, FILE_PATTERNS
    )
except ImportError:
    from config import (
        IMAGES_DIR, MASKS_DIR, PROCESSED_DATA_DIR, 
        DATA_CONFIG, TRAINING_CONFIG, FILE_PATTERNS
    )

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BaseDataLoader(ABC):
    """
    Base data loader class for hair segmentation dataset.
    
    Contains common functionality for loading, preprocessing, and managing data.
    """
    
    def __init__(self, 
                 images_dir: Path = IMAGES_DIR,
                 masks_dir: Path = MASKS_DIR,
                 processed_dir: Path = PROCESSED_DATA_DIR,
                 image_size: Tuple[int, int] = DATA_CONFIG["image_size"],
                 normalization_factor: float = DATA_CONFIG["normalization_factor"]):
        """
        Initialize the base data loader.
        
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
        
        # File paths storage
        self.image_paths = []
        self.mask_paths = []
        
    def get_file_paths(self) -> Tuple[List[str], List[str]]:
        """
        Get all image and mask file paths.
        
        Returns:
            Tuple of (image_paths, mask_paths)
        """
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
        
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        
        return image_paths, mask_paths
        
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
    
    @abstractmethod
    def load_data(self):
        """
        Abstract method for loading data.
        Must be implemented by subclasses.
        """
        pass
    
    @abstractmethod
    def split_data(self, validation_split: float = TRAINING_CONFIG["validation_split"],
                   random_seed: int = TRAINING_CONFIG["random_seed"]):
        """
        Abstract method for splitting data.
        Must be implemented by subclasses.
        """
        pass
    
    def get_data_info(self) -> dict:
        """
        Get information about the dataset.
        
        Returns:
            Dictionary containing dataset information
        """
        info = {
            "image_size": self.image_size,
            "normalization_factor": self.normalization_factor,
            "images_dir_exists": self.images_dir.exists(),
            "masks_dir_exists": self.masks_dir.exists(),
            "processed_dir_exists": self.processed_dir.exists(),
            "total_file_paths": len(self.image_paths)
        }
        
        return info
