"""
PyTorch Dataset for hair segmentation with traditional loading.
Handles pre-loaded images and masks with optional augmentation.
"""

import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Tuple
import logging

from model.utils.augmentation import Augmentation

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TraditionalDataset(Dataset):
    """
    PyTorch Dataset for traditional loading with augmentation support.
    """
    
    def __init__(self, 
                images: np.ndarray, 
                masks: np.ndarray, 
                use_augmentation: bool = False,
                is_training: bool = True):
        """
        Initialize the dataset.
        
        Args:
            images: Numpy array of images
            masks: Numpy array of masks
            use_augmentation: Whether to apply augmentations
            is_training: Whether this is a training dataset
        """
        self.images = images
        self.masks = masks
        self.use_augmentation = use_augmentation and is_training
        self.is_training = is_training
        
        if len(images) != len(masks):
            raise ValueError(f"Number of images ({len(images)}) doesn't match number of masks ({len(masks)})")
            
        logger.info(f"Created {'training' if is_training else 'validation'} dataset with {len(images)} samples")
        if self.use_augmentation:
            logger.info(f"Data augmentation enabled for {'training' if is_training else 'validation'} dataset")
    
    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.images)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a single image-mask pair.
        
        Args:
            idx: Index of the sample
            
        Returns:
            Tuple of (image, mask) as torch tensors
        """
        # Get image and mask
        image = self.images[idx].copy()
        mask = self.masks[idx].copy()
        
        # Apply augmentations if enabled
        if self.use_augmentation:
            image, mask = Augmentation.apply_augmentations(image, mask)
        
        # Convert to PyTorch tensors
        # Image: (H, W, C) -> (C, H, W)
        image = torch.FloatTensor(image).permute(2, 0, 1)
        # Mask: (H, W) -> (1, H, W)
        mask = torch.FloatTensor(mask).unsqueeze(0)
        
        return image, mask 