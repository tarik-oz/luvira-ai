"""
Enhanced PyTorch Dataset for hair segmentation with augmentations.
Handles on-demand loading of images and masks and applies data augmentation.
"""

import cv2
import numpy as np
import torch
import random
from torch.utils.data import Dataset
from typing import List, Tuple
import logging

try:
    from .config import DATA_CONFIG
except ImportError:
    from config import DATA_CONFIG

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class KaggleLazyDataset(Dataset):
    """
    PyTorch Dataset for hair segmentation that loads images on-demand and applies augmentations.
    This prevents loading all images into memory at once and enhances training with data augmentation.
    """
    
    def __init__(self, 
                image_paths: List[str], 
                mask_paths: List[str], 
                image_size: Tuple[int, int] = DATA_CONFIG["image_size"],
                normalization_factor: float = DATA_CONFIG["normalization_factor"],
                apply_augmentations: bool = True,
                is_training: bool = True):
        """
        Initialize the dataset.
        
        Args:
            image_paths: List of image file paths
            mask_paths: List of mask file paths
            image_size: Target size for images (height, width)
            normalization_factor: Factor to normalize pixel values
            apply_augmentations: Whether to apply augmentations
            is_training: Whether this is a training dataset (augmentations only applied during training)
        """
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.image_size = image_size
        self.normalization_factor = normalization_factor
        self.enable_augmentations = apply_augmentations and is_training
        self.is_training = is_training
        
        if len(image_paths) != len(mask_paths):
            raise ValueError(f"Number of images ({len(image_paths)}) doesn't match number of masks ({len(mask_paths)})")
        
        logger.info(f"Created {'training' if is_training else 'validation'} dataset with {len(image_paths)} samples")
        logger.info(f"Data augmentation enabled: {self.enable_augmentations}")
    
    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.image_paths)
    
    def _apply_horizontal_flip(self, image: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Apply horizontal flip augmentation."""
        if random.random() < 0.5:
            image = cv2.flip(image, 1)  # 1 means horizontal flip
            mask = cv2.flip(mask, 1)
        return image, mask
    
    def _apply_random_rotation(self, image: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Apply random rotation augmentation (±10°)."""
        if random.random() < 0.5:
            angle = random.uniform(-10, 10)
            h, w = image.shape[:2]
            center = (w/2, h/2)
            
            # Get rotation matrix
            rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
            
            # Apply rotation
            image = cv2.warpAffine(image, rotation_matrix, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
            mask = cv2.warpAffine(mask, rotation_matrix, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        
        return image, mask
    
    def _apply_color_jitter(self, image: np.ndarray) -> np.ndarray:
        """Apply color jitter augmentation (brightness, contrast, saturation)."""
        if random.random() < 0.5:
            # Convert to HSV for better color manipulation
            hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
            
            # Random saturation change
            s_factor = random.uniform(0.7, 1.3)
            hsv_image[:, :, 1] = np.clip(hsv_image[:, :, 1] * s_factor, 0, 255)
            
            # Random hue change
            h_shift = random.uniform(-10, 10)
            hsv_image[:, :, 0] = np.clip(hsv_image[:, :, 0] + h_shift, 0, 179)
            
            # Convert back to BGR
            image = cv2.cvtColor(hsv_image.astype(np.uint8), cv2.COLOR_HSV2BGR)
        
        return image
    
    def _apply_random_brightness_contrast(self, image: np.ndarray) -> np.ndarray:
        """Apply random brightness and contrast augmentation."""
        if random.random() < 0.5:
            # Brightness
            brightness = random.uniform(0.8, 1.2)
            image = cv2.convertScaleAbs(image, alpha=brightness, beta=0)
            
            # Contrast
            contrast = random.uniform(0.8, 1.2)
            image = cv2.convertScaleAbs(image, alpha=contrast, beta=128*(1-contrast))
        
        return image
    
    def _apply_elastic_transform(self, image: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Apply elastic transform augmentation (with mild settings for hair)."""
        if random.random() < 0.3:  # Lower probability for elastic transform
            # Parameters for mild elastic transform
            alpha = random.uniform(30, 60)  # Lower alpha for milder effect
            sigma = random.uniform(4, 6)
            
            h, w = image.shape[:2]
            
            # Generate random displacement fields
            dx = np.random.rand(h, w).astype(np.float32) * 2 - 1
            dy = np.random.rand(h, w).astype(np.float32) * 2 - 1
            
            # Gaussian blur for smooth displacement
            dx = cv2.GaussianBlur(dx, (0, 0), sigma)
            dy = cv2.GaussianBlur(dy, (0, 0), sigma)
            
            # Normalize and scale
            dx = dx * alpha
            dy = dy * alpha
            
            # Create mesh grid
            x, y = np.meshgrid(np.arange(w), np.arange(h))
            
            # Displacement
            map_x = np.float32(x + dx)
            map_y = np.float32(y + dy)
            
            # Apply displacement
            image = cv2.remap(image, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
            mask = cv2.remap(mask, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        
        return image, mask
    
    def apply_augmentations(self, image: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Apply all augmentations in sequence."""
        if self.enable_augmentations:
            # Apply augmentations that affect both image and mask
            image, mask = self._apply_horizontal_flip(image, mask)
            image, mask = self._apply_random_rotation(image, mask)
            image, mask = self._apply_elastic_transform(image, mask)
            
            # Apply augmentations that only affect the image
            image = self._apply_color_jitter(image)
            image = self._apply_random_brightness_contrast(image)
        
        return image, mask
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Load and return a single image-mask pair with optional augmentations.
        
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
        
        # Apply augmentations before resizing
        image, mask = self.apply_augmentations(image, mask)
        
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