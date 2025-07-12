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

try:
    from ..config import DATA_CONFIG
except ImportError:
    from config import DATA_CONFIG

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HairSegmentationDataset(Dataset):
    """
    PyTorch Dataset for hair segmentation that loads images on-demand.
    This prevents loading all images into memory at once.
    """
    
    def __init__(self, 
                image_paths: List[str], 
                mask_paths: List[str], 
                image_size: Tuple[int, int] = DATA_CONFIG["image_size"],
                normalization_factor: float = DATA_CONFIG["normalization_factor"],
                lazy_loading: bool = True):
        """
        Initialize the dataset.
        
        Args:
            image_paths: List of image file paths
            mask_paths: List of mask file paths
            image_size: Target size for images (height, width)
            normalization_factor: Factor to normalize pixel values
            lazy_loading: Whether to use lazy loading (always True for this dataset)
        """
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.image_size = image_size
        self.normalization_factor = normalization_factor
        
        if len(image_paths) != len(mask_paths):
            raise ValueError(f"Number of images ({len(image_paths)}) doesn't match number of masks ({len(mask_paths)})")
    
    def __len__(self):
        """Return the number of samples in the dataset."""
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        """
        Load and return a single image-mask pair.
        
        Args:
            idx: Index of the sample
            
        Returns:
            Tuple of (image, mask) as torch tensors
        """
        # Load image
        image = cv2.imread(self.image_paths[idx], cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Could not load image: {self.image_paths[idx]}")
        
        # Load mask
        mask = cv2.imread(self.mask_paths[idx], cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise ValueError(f"Could not load mask: {self.mask_paths[idx]}")
        
        # Resize
        image = cv2.resize(image, self.image_size)
        mask = cv2.resize(mask, self.image_size)
        
        # Normalize
        image = image.astype(np.float32) / self.normalization_factor
        mask = mask.astype(np.float32) / 255.0
        
        # Convert to torch tensors
        # Image: HWC -> CHW
        image = torch.from_numpy(image).permute(2, 0, 1)
        # Mask: HW -> 1HW
        mask = torch.from_numpy(mask).unsqueeze(0)
        
        return image, mask
