"""
PyTorch Dataset for hair segmentation.
Handles on-demand loading of images and masks.
"""

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import List, Tuple
import logging

from model.config import DATA_CONFIG
from model.utils.augmentation import Augmentation

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LazyDataset(Dataset):
    """
    PyTorch Dataset for hair segmentation that loads images on-demand.
    This prevents loading all images into memory at once.
    """
    
    def __init__(self, 
                image_paths: List[str], 
                mask_paths: List[str], 
                image_size: Tuple[int, int] = DATA_CONFIG["image_size"],
                normalization_factor: float = DATA_CONFIG["normalization_factor"],
                use_augmentation: bool = False,
                is_training: bool = True):
        """
        Initialize the dataset.
        
        Args:
            image_paths: List of image file paths
            mask_paths: List of mask file paths
            image_size: Target size for images (height, width)
            normalization_factor: Factor to normalize pixel values
            use_augmentation: Whether to apply augmentations
            is_training: Whether this is a training dataset
        """
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.image_size = image_size
        self.normalization_factor = normalization_factor
        self.use_augmentation = use_augmentation and is_training
        self.is_training = is_training
        
        if len(image_paths) != len(mask_paths):
            raise ValueError(f"Number of images ({len(image_paths)}) doesn't match number of masks ({len(mask_paths)})")
            
        logger.info(f"Created {'training' if is_training else 'validation'} dataset with {len(image_paths)} samples")
        if self.use_augmentation:
            logger.info(f"Data augmentation enabled for {'training' if is_training else 'validation'} dataset")
    
    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Load and return a single image-mask pair.
        
        Args:
            idx: Index of the sample
            
        Returns:
            Tuple of (image, mask) as torch tensors
        """
        # Load image
        image = cv2.imread(str(self.image_paths[idx]), cv2.IMREAD_COLOR)
        if image is None:
            logger.error(f"Could not load image: {self.image_paths[idx]}")
            raise ValueError(f"Could not load image: {self.image_paths[idx]}")
        
        # Load mask
        mask = cv2.imread(str(self.mask_paths[idx]), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            logger.error(f"Could not load mask: {self.mask_paths[idx]}")
            raise ValueError(f"Could not load mask: {self.mask_paths[idx]}")
        
        # Apply augmentations if enabled
        if self.use_augmentation:
            image, mask = Augmentation.apply_augmentations(image, mask)
        
        # Resize
        image = cv2.resize(image, self.image_size)
        mask = cv2.resize(mask, self.image_size)
        
        # Normalize
        image = image.astype(np.float32) / self.normalization_factor
        mask = mask.astype(np.float32) / 255.0
        
        # Convert to PyTorch tensors
        # Image: (H, W, C) -> (C, H, W)
        image = torch.FloatTensor(image).permute(2, 0, 1)
        # Mask: (H, W) -> (1, H, W)
        mask = torch.FloatTensor(mask).unsqueeze(0)
        
        return image, mask
 