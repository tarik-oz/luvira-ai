"""
File validation utilities for Hair Segmentation API
"""

import logging
import cv2
import numpy as np
from typing import Optional, Tuple
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