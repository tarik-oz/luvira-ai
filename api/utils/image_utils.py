"""
Image utility functions for common operations across services
"""

import logging
from pathlib import Path
from typing import Optional
import cv2
import numpy as np

from ..core.exceptions import ImageProcessingException
from .validators import FileValidator

logger = logging.getLogger(__name__)


class ImageUtils:
    """Utility class for common image operations"""
    
    @staticmethod
    def process_uploaded_file(file) -> np.ndarray:
        """
        Process uploaded file and return as numpy array
        
        Args:
            file: Uploaded file object
            
        Returns:
            Processed image as numpy array
        """
        try:
            # Read file data
            image_data = file.file.read()
            file.file.seek(0)  # Reset file pointer
            
            return ImageUtils.bytes_to_image(image_data)
            
        except Exception as e:
            logger.error(f"Failed to process uploaded file: {str(e)}")
            raise ImageProcessingException(f"Failed to process uploaded file: {str(e)}")
    
    @staticmethod
    def bytes_to_image(image_data: bytes) -> np.ndarray:
        """
        Convert image bytes to numpy array
        
        Args:
            image_data: Image data as bytes
            
        Returns:
            Image as numpy array
        """
        try:
            # Convert bytes to numpy array
            nparr = np.frombuffer(image_data, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if image is None:
                raise ImageProcessingException("Could not decode image")
            
            # Validate image
            FileValidator.validate_image_dimensions(image)
            FileValidator.validate_image_format(image)
            
            return image
            
        except Exception as e:
            logger.error(f"Image bytes processing failed: {str(e)}")
            raise ImageProcessingException(f"Image bytes processing failed: {str(e)}")
    
    @staticmethod
    def bytes_to_mask(mask_bytes: bytes) -> np.ndarray:
        """
        Convert mask bytes to numpy array
        
        Args:
            mask_bytes: Mask data as bytes
            
        Returns:
            Mask as numpy array
        """
        try:
            nparr = np.frombuffer(mask_bytes, np.uint8)
            mask = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
            
            if mask is None:
                raise ImageProcessingException("Could not decode mask")
            
            return mask
            
        except Exception as e:
            logger.error(f"Mask bytes processing failed: {str(e)}")
            raise ImageProcessingException(f"Mask bytes processing failed: {str(e)}")
    
    @staticmethod
    def image_to_bytes(image: np.ndarray, format_type: str = 'png', is_rgb: bool = False) -> bytes:
        """
        Convert image array to bytes
        
        Args:
            image: Image as numpy array
            format_type: Output format ('png', 'jpg')
            is_rgb: Whether image is in RGB format (needs BGR conversion)
            
        Returns:
            Image as bytes
        """
        try:
            # Convert RGB to BGR if needed
            if is_rgb:
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
            # Encode image
            _, buffer = cv2.imencode(f'.{format_type}', image)
            return buffer.tobytes()
            
        except Exception as e:
            logger.error(f"Failed to convert image to bytes: {str(e)}")
            raise ImageProcessingException(f"Failed to convert image to bytes: {str(e)}")
    
    @staticmethod
    def mask_to_bytes(mask: np.ndarray, format_type: str = 'png') -> bytes:
        """
        Convert mask array to bytes
        
        Args:
            mask: Mask as numpy array (values 0-1 or 0-255)
            format_type: Output format ('png', 'jpg')
            
        Returns:
            Mask as bytes
        """
        try:
            # Ensure mask is in 0-255 range
            if mask.max() <= 1.0:
                mask_255 = (mask * 255).astype(np.uint8)
            else:
                mask_255 = mask.astype(np.uint8)
            
            # Encode mask
            _, buffer = cv2.imencode(f'.{format_type}', mask_255)
            return buffer.tobytes()
            
        except Exception as e:
            logger.error(f"Failed to convert mask to bytes: {str(e)}")
            raise ImageProcessingException(f"Failed to convert mask to bytes: {str(e)}")
    
    @staticmethod
    def cleanup_temp_file(temp_path: Optional[str]) -> None:
        """
        Clean up temporary file
        
        Args:
            temp_path: Path to temporary file
        """
        if temp_path:
            try:
                Path(temp_path).unlink(missing_ok=True)
            except Exception as e:
                logger.warning(f"Failed to cleanup temp file {temp_path}: {str(e)}")
