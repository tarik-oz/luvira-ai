"""
Data augmentation utilities for hair segmentation.
Contains augmentation functions that can be used with any data loader.
"""

import cv2
import numpy as np
import random
from typing import Tuple

class Augmentation:
    """
    Augmentation utilities for hair segmentation.
    
    Contains methods for various image augmentations that can be applied
    to both images and masks.
    """
    
    @staticmethod
    def horizontal_flip(image: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Apply horizontal flip augmentation."""
        if random.random() < 0.5:
            image = cv2.flip(image, 1)  # 1 means horizontal flip
            mask = cv2.flip(mask, 1)
        return image, mask
    
    @staticmethod
    def random_rotation(image: np.ndarray, mask: np.ndarray, max_angle: float = 10.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply random rotation augmentation.
        
        Args:
            image: Input image
            mask: Input mask
            max_angle: Maximum rotation angle in degrees
            
        Returns:
            Tuple of (rotated_image, rotated_mask)
        """
        if random.random() < 0.5:
            # Generate random angle
            angle = random.uniform(-max_angle, max_angle)
            
            # Get image dimensions
            height, width = image.shape[:2]
            center = (width // 2, height // 2)
            
            # Calculate rotation matrix
            rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
            
            # Apply rotation
            image = cv2.warpAffine(
                image, rotation_matrix, (width, height),
                flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT
            )
            mask = cv2.warpAffine(
                mask, rotation_matrix, (width, height),
                flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_REFLECT
            )
            
        return image, mask
    
    @staticmethod
    def color_jitter(image: np.ndarray, hue_shift: float = 0.05, sat_shift: float = 0.1, val_shift: float = 0.1) -> np.ndarray:
        """
        Apply color jitter augmentation (only to image, not mask).
        
        Args:
            image: Input image (BGR format)
            hue_shift: Maximum hue shift (0-1)
            sat_shift: Maximum saturation shift (0-1)
            val_shift: Maximum value shift (0-1)
            
        Returns:
            Color jittered image
        """
        if random.random() < 0.5:
            # Convert to HSV
            hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
            
            # Random shifts
            h_shift = random.uniform(-hue_shift, hue_shift) * 180
            s_shift = random.uniform(-sat_shift, sat_shift) * 255
            v_shift = random.uniform(-val_shift, val_shift) * 255
            
            # Apply shifts
            hsv_image[..., 0] = (hsv_image[..., 0] + h_shift) % 180
            hsv_image[..., 1] = np.clip(hsv_image[..., 1] + s_shift, 0, 255)
            hsv_image[..., 2] = np.clip(hsv_image[..., 2] + v_shift, 0, 255)
            
            # Convert back to BGR
            image = cv2.cvtColor(hsv_image.astype(np.uint8), cv2.COLOR_HSV2BGR)
            
        return image
    
    @staticmethod
    def random_brightness_contrast(image: np.ndarray, 
                                  brightness_limit: float = 0.2, 
                                  contrast_limit: float = 0.2) -> np.ndarray:
        """
        Apply random brightness and contrast augmentation (only to image, not mask).
        
        Args:
            image: Input image
            brightness_limit: Maximum brightness shift (0-1)
            contrast_limit: Maximum contrast shift (0-1)
            
        Returns:
            Augmented image
        """
        if random.random() < 0.5:
            # Random brightness
            alpha = 1.0 + random.uniform(-contrast_limit, contrast_limit)
            beta = random.uniform(-brightness_limit, brightness_limit) * 255
            
            # Apply transformation: new_pixel = alpha * old_pixel + beta
            image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
            
        return image
    
    @staticmethod
    def apply_augmentations(image: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply all augmentations in sequence.
        
        Args:
            image: Input image
            mask: Input mask
            
        Returns:
            Tuple of (augmented_image, augmented_mask)
        """
        # Apply augmentations that affect both image and mask
        image, mask = Augmentation.horizontal_flip(image, mask)
        image, mask = Augmentation.random_rotation(image, mask)
        
        # Apply augmentations that only affect the image
        image = Augmentation.color_jitter(image)
        image = Augmentation.random_brightness_contrast(image)
        
        return image, mask 