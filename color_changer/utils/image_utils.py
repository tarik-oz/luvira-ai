"""
Image utilities for hair color change operations.
"""

import cv2
import numpy as np
from typing import Optional


class ImageUtils:
    """
    Utility functions for image processing.
    """
    
    @staticmethod
    def load_image(path: str, grayscale: bool = False) -> Optional[np.ndarray]:
        """
        Load image from path.
        
        Args:
            path: Path to image file
            grayscale: Whether to load as grayscale
            
        Returns:
            Image as numpy array or None if failed
        """
        try:
            if grayscale:
                image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            else:
                image = cv2.imread(path)
            
            if image is None:
                print(f"Failed to load image from {path}")
            return image
        except Exception as e:
            print(f"Error loading image from {path}: {str(e)}")
            return None
    
    @staticmethod
    def save_image(image: np.ndarray, path: str, convert_to_bgr: bool = False) -> bool:
        """
        Save image to path.
        
        Args:
            image: Image as numpy array
            path: Path to save image
            convert_to_bgr: Whether to convert from RGB to BGR before saving
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Convert to BGR if needed (OpenCV uses BGR format for saving)
            if convert_to_bgr:
                image_to_save = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            else:
                image_to_save = image
                
            success = cv2.imwrite(path, image_to_save)
            if not success:
                print(f"Failed to save image to {path}")
            return success
        except Exception as e:
            print(f"Error saving image to {path}: {str(e)}")
            return False
