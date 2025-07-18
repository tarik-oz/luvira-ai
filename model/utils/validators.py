"""
Validation utilities for model inference.
"""

from pathlib import Path
from typing import List, Optional

class ImageValidator:
    """
    Validator class for image-related operations.
    """
    
    @staticmethod
    def validate_input_directory(input_dir: Path, default_dir: Optional[Path] = None) -> Path:
        """
        Validate input directory exists.
        
        Args:
            input_dir: Directory to validate
            default_dir: Default directory to use if input_dir is None
            
        Returns:
            Validated Path object
            
        Raises:
            ValueError: If directory doesn't exist
        """
        # Use default_dir if input_dir is None
        search_dir = input_dir if input_dir else default_dir
        
        # Check directory exists
        if not search_dir.exists():
            raise ValueError(f"Images directory not found: {search_dir}")
            
        return search_dir
    
    @staticmethod
    def validate_image_names(image_names: List[str], input_dir: Path) -> None:
        """
        Validate that all specified images exist in the directory.
        
        Args:
            image_names: List of image names to validate
            input_dir: Directory to look for images in
            
        Raises:
            ValueError: If any images are not found
        """
        missing_images = []
        for img_name in image_names:
            if not (input_dir / img_name).exists():
                missing_images.append(img_name)
        if missing_images:
            raise ValueError(f"Images not found in {input_dir}: {', '.join(missing_images)}")
    
    @staticmethod
    def find_valid_images(input_dir: Path) -> List[str]:
        """
        Find all valid images in a directory.
        
        Args:
            input_dir: Directory to search in
            
        Returns:
            List of valid image names
            
        Raises:
            ValueError: If no valid images found
        """
        valid_extensions = (".jpg", ".jpeg", ".png")
        image_files = [
            f.name for f in input_dir.glob("*") 
            if f.suffix.lower() in valid_extensions
        ]
        
        if not image_files:
            raise ValueError(f"No valid images found in {input_dir}")
            
        return image_files 