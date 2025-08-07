"""
File validation utilities for Hair Segmentation API
"""

import logging
import numpy as np
from typing import Optional, List
from fastapi import UploadFile
from pathlib import Path

from ..config import FILE_VALIDATION
from ..core.exceptions import FileValidationException


class FileValidator:
    """Advanced file validation for image uploads"""
    
    @staticmethod
    def validate_upload_file(file: UploadFile) -> None:
        """
        Comprehensive file validation
        
        Args:
            file: Uploaded file to validate
            
        Raises:
            FileValidationException: If validation fails
        """
        # Check if file exists
        if not file:
            raise FileValidationException("No file provided")
        
        # Validate filename
        FileValidator._validate_filename(file.filename)
        
        # Validate content type
        FileValidator._validate_content_type(file.content_type)
        
        # Validate file size
        FileValidator._validate_file_size(file.size)
    
    @staticmethod
    def _validate_filename(filename: Optional[str]) -> None:
        """Validate filename"""
        if not filename:
            raise FileValidationException("Filename is required")
        
        if len(filename) > FILE_VALIDATION["max_filename_length"]:
            raise FileValidationException(
                f"Filename too long. Maximum length: {FILE_VALIDATION['max_filename_length']} characters"
            )
        
        # Check for invalid characters
        invalid_chars = ['<', '>', ':', '"', '|', '?', '*', '\\', '/']
        if any(char in filename for char in invalid_chars):
            raise FileValidationException("Filename contains invalid characters")
    
    @staticmethod
    def _validate_content_type(content_type: Optional[str]) -> None:
        """Validate content type"""
        if not content_type:
            raise FileValidationException("Content type is required")
        
        if content_type not in FILE_VALIDATION["allowed_image_types"]:
            raise FileValidationException(
                f"File type not allowed. Allowed types: {FILE_VALIDATION['allowed_image_types']}"
            )
    
    @staticmethod
    def _validate_file_size(file_size: Optional[int]) -> None:
        """Validate file size"""
        if file_size is None:
            return  # Allow if size is unknown
        
        if file_size <= 0:
            raise FileValidationException("File size must be greater than 0")
        
        if file_size > FILE_VALIDATION["max_file_size"]:
            max_size_mb = FILE_VALIDATION["max_file_size"] // (1024 * 1024)
            raise FileValidationException(f"File too large. Maximum size: {max_size_mb}MB")
    
    @staticmethod
    def validate_image_dimensions(image: np.ndarray) -> None:
        """
        Validate image dimensions
        
        Args:
            image: Image as numpy array
            
        Raises:
            FileValidationException: If dimensions are invalid
        """
        if image is None:
            raise FileValidationException("Invalid image data")
        
        height, width = image.shape[:2]
        min_width, min_height = FILE_VALIDATION["min_image_dimensions"]
        max_width, max_height = FILE_VALIDATION["max_image_dimensions"]
        
        if width < min_width or height < min_height:
            raise FileValidationException(
                f"Image too small. Minimum dimensions: {min_width}x{min_height}"
            )
        
        if width > max_width or height > max_height:
            raise FileValidationException(
                f"Image too large. Maximum dimensions: {max_width}x{max_height}"
            )
        
        # Log validation success
        logger = logging.getLogger(__name__)
        logger.debug(f"Image dimensions validated: {width}x{height}")
    
    @staticmethod
    def validate_image_format(image: np.ndarray) -> None:
        """
        Validate image format and channels
        
        Args:
            image: Image as numpy array
            
        Raises:
            FileValidationException: If format is invalid
        """
        if image is None:
            raise FileValidationException("Invalid image data")
        
        # Check if image has valid shape
        if len(image.shape) not in [2, 3]:
            raise FileValidationException("Invalid image format")
        
        # Check if color image has 3 channels
        if len(image.shape) == 3 and image.shape[2] != 3:
            raise FileValidationException("Image must be RGB (3 channels) or grayscale")


class ModelPathValidator:
    """Model path validation utilities"""
    
    @staticmethod
    def validate_model_path(model_path: str) -> None:
        """
        Validate model file path
        
        Args:
            model_path: Path to model file
            
        Raises:
            FileValidationException: If path is invalid
        """
        if not model_path:
            raise FileValidationException("Model path is required")
        
        path = Path(model_path)
        
        if not path.exists():
            raise FileValidationException(f"Model file not found: {model_path}")
        
        if not path.is_file():
            raise FileValidationException(f"Path is not a file: {model_path}")
        
        # Check file extension
        if path.suffix.lower() not in ['.pth', '.pt', '.ckpt']:
            raise FileValidationException(
                f"Invalid model file extension. Supported: .pth, .pt, .ckpt"
            )


class ColorValidator:
    """Color validation utilities"""
    
    @staticmethod
    def validate_color_name(color_name: str) -> None:
        """
        Validate color name from COLORS config (case-insensitive)
        
        Args:
            color_name: Color name to validate
            
        Raises:
            FileValidationException: If color name is invalid
        """
        try:
            # Import here to avoid circular import
            import sys
            import os
            sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'color_changer'))
            from color_changer.config.color_config import COLORS
            
            valid_colors = [name for _, name in COLORS]
            # Case-insensitive comparison
            valid_colors_lower = [color.lower() for color in valid_colors]
            
            if color_name.lower() not in valid_colors_lower:
                raise FileValidationException(
                    f"Invalid color name '{color_name}'. Available colors: {valid_colors}"
                )
        except ImportError as e:
            raise FileValidationException(f"Could not load color configuration: {str(e)}")
    
    @staticmethod
    def get_correct_color_name(color_name: str) -> str:
        """
        Get the correct case color name from user input
        
        Args:
            color_name: Color name from user (any case)
            
        Returns:
            Correct case color name
        """
        try:
            import sys
            import os
            sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'color_changer'))
            from color_changer.config.color_config import COLORS
            
            valid_colors = [name for _, name in COLORS]
            # Find matching color (case-insensitive)
            for color in valid_colors:
                if color.lower() == color_name.lower():
                    return color
            
            # If not found, return original (validation will catch this)
            return color_name
        except ImportError:
            return color_name
    
    @staticmethod
    def validate_tone_name(color_name: str, tone_name: str) -> None:
        """
        Validate tone name for a specific color (case-insensitive)
        
        Args:
            color_name: Base color name
            tone_name: Tone name to validate
            
        Raises:
            FileValidationException: If tone name is invalid
        """
        try:
            # Import here to avoid circular import
            import sys
            import os
            sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'color_changer'))
            from color_changer.config.color_config import CUSTOM_TONES
            
            # Get correct case color name
            correct_color_name = ColorValidator.get_correct_color_name(color_name)
            
            if correct_color_name not in CUSTOM_TONES:
                raise FileValidationException(f"No tones available for color '{color_name}'")
            
            valid_tones = list(CUSTOM_TONES[correct_color_name].keys())
            # Case-insensitive comparison
            valid_tones_lower = [tone.lower() for tone in valid_tones]
            
            if tone_name.lower() not in valid_tones_lower:
                raise FileValidationException(
                    f"Invalid tone name '{tone_name}' for color '{color_name}'. Available tones: {valid_tones}"
                )
        except ImportError as e:
            raise FileValidationException(f"Could not load color configuration: {str(e)}")
    
    @staticmethod
    def get_correct_tone_name(color_name: str, tone_name: str) -> str:
        """
        Get the correct case tone name from user input
        
        Args:
            color_name: Color name
            tone_name: Tone name from user (any case)
            
        Returns:
            Correct case tone name
        """
        try:
            import sys
            import os
            sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'color_changer'))
            from color_changer.config.color_config import CUSTOM_TONES
            
            # Get correct case color name
            correct_color_name = ColorValidator.get_correct_color_name(color_name)
            
            if correct_color_name in CUSTOM_TONES:
                valid_tones = list(CUSTOM_TONES[correct_color_name].keys())
                # Find matching tone (case-insensitive)
                for tone in valid_tones:
                    if tone.lower() == tone_name.lower():
                        return tone
            
            # If not found, return original (validation will catch this)
            return tone_name
        except ImportError:
            return tone_name
    
    @staticmethod
    def validate_rgb_color(color_list: List[int]) -> None:
        """
        Validate RGB color values (kept for backward compatibility)
        
        Args:
            color_list: List of RGB color values [R, G, B]
            
        Raises:
            FileValidationException: If color values are invalid
        """
        if not isinstance(color_list, list):
            raise FileValidationException("Target color must be a list")
        
        if len(color_list) != 3:
            raise FileValidationException("Target color must have exactly 3 values [R, G, B]")
        
        for i, color in enumerate(color_list):
            if not isinstance(color, int):
                raise FileValidationException(f"Color value at index {i} must be an integer")
            
            if color < 0 or color > 255:
                raise FileValidationException(f"Color value at index {i} must be between 0 and 255") 