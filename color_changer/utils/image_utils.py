"""
Image utilities for hair color change operations.
"""

import cv2
import numpy as np
from typing import Tuple, Optional, Union


class ImageUtils:
    """
    Utility functions for image processing.
    """
    
    @staticmethod
    def load_image(path: str) -> Optional[np.ndarray]:
        """
        Load image from path.
        
        Args:
            path: Path to image file
            
        Returns:
            Image as numpy array or None if failed
        """
        try:
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
    
    @staticmethod
    def resize_image(image: np.ndarray, max_dimension: int = 800) -> np.ndarray:
        """
        Resize image to have maximum dimension of max_dimension while preserving aspect ratio.
        
        Args:
            image: Image to resize
            max_dimension: Maximum dimension (width or height)
            
        Returns:
            Resized image
        """
        # Get image dimensions
        height, width = image.shape[:2]
        
        # Calculate aspect ratio
        aspect_ratio = width / height
        
        # Resize based on max dimension
        if width > height and width > max_dimension:
            new_width = max_dimension
            new_height = int(new_width / aspect_ratio)
        elif height >= width and height > max_dimension:
            new_height = max_dimension
            new_width = int(new_height * aspect_ratio)
        else:
            # No need to resize
            return image
        
        # Resize image
        resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
        return resized
    
    @staticmethod
    def convert_color_space(
        image: np.ndarray, 
        source_space: str, 
        target_space: str
    ) -> np.ndarray:
        """
        Convert image between color spaces.
        
        Args:
            image: Image to convert
            source_space: Source color space ('RGB', 'BGR', 'HSV', 'Lab', etc.)
            target_space: Target color space ('RGB', 'BGR', 'HSV', 'Lab', etc.)
            
        Returns:
            Converted image
        """
        # Define conversion mappings
        color_space_map = {
            'BGR2RGB': cv2.COLOR_BGR2RGB,
            'RGB2BGR': cv2.COLOR_RGB2BGR,
            'BGR2HSV': cv2.COLOR_BGR2HSV,
            'HSV2BGR': cv2.COLOR_HSV2BGR,
            'RGB2HSV': cv2.COLOR_RGB2HSV,
            'HSV2RGB': cv2.COLOR_HSV2RGB,
            'BGR2Lab': cv2.COLOR_BGR2Lab,
            'Lab2BGR': cv2.COLOR_Lab2BGR,
            'RGB2Lab': cv2.COLOR_RGB2Lab,
            'Lab2RGB': cv2.COLOR_Lab2RGB,
        }
        
        # Create conversion code
        conversion_key = f"{source_space}2{target_space}"
        if conversion_key in color_space_map:
            return cv2.cvtColor(image, color_space_map[conversion_key])
        else:
            # Try two-step conversion through BGR
            step1_key = f"{source_space}2BGR"
            step2_key = f"BGR2{target_space}"
            
            if step1_key in color_space_map and step2_key in color_space_map:
                intermediate = cv2.cvtColor(image, color_space_map[step1_key])
                return cv2.cvtColor(intermediate, color_space_map[step2_key])
            else:
                raise ValueError(f"Unsupported color space conversion: {source_space} to {target_space}")
    
    @staticmethod
    def create_mask_overlay(
        image: np.ndarray, 
        mask: np.ndarray, 
        color: Tuple[int, int, int] = (0, 255, 0), 
        alpha: float = 0.5
    ) -> np.ndarray:
        """
        Create overlay of mask on image for visualization.
        
        Args:
            image: Original image (BGR or RGB)
            mask: Binary or grayscale mask
            color: Color for overlay (BGR or RGB matching image)
            alpha: Opacity of overlay (0-1)
            
        Returns:
            Image with mask overlay
        """
        # Ensure mask is properly scaled to 0-255
        if mask.dtype != np.uint8:
            mask_viz = (mask * 255).astype(np.uint8)
        else:
            mask_viz = mask.copy()
            
        # Create color mask
        if len(image.shape) == 3 and image.shape[2] == 3:
            h, w = mask_viz.shape[:2]
            color_mask = np.zeros((h, w, 3), dtype=np.uint8)
            color_mask[:] = color
        else:
            raise ValueError("Image must be a 3-channel color image")
            
        # Create overlay
        mask_binary = mask_viz > 127
        overlay = image.copy()
        overlay[mask_binary] = cv2.addWeighted(
            image[mask_binary], 1 - alpha, 
            color_mask[mask_binary], alpha, 0
        )
        
        return overlay 