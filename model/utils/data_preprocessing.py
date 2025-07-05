"""
Data preprocessing utilities for hair segmentation dataset.
Handles data preparation and validation.
"""

import os
import glob
from pathlib import Path
from typing import List, Tuple
import logging

from config import IMAGES_DIR, MASKS_DIR, FILE_PATTERNS

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def validate_dataset(images_dir: Path = IMAGES_DIR, 
                    masks_dir: Path = MASKS_DIR) -> Tuple[bool, dict]:
    """
    Validate the dataset structure and files.
    
    Args:
        images_dir: Directory containing images
        masks_dir: Directory containing masks
        
    Returns:
        Tuple of (is_valid, validation_info)
    """
    validation_info = {
        "images_dir_exists": False,
        "masks_dir_exists": False,
        "image_files": [],
        "mask_files": [],
        "matching_pairs": 0,
        "errors": []
    }
    
    # Check if directories exist
    if not images_dir.exists():
        validation_info["errors"].append(f"Images directory does not exist: {images_dir}")
    else:
        validation_info["images_dir_exists"] = True
    
    if not masks_dir.exists():
        validation_info["errors"].append(f"Masks directory does not exist: {masks_dir}")
    else:
        validation_info["masks_dir_exists"] = True
    
    # Get image files
    if validation_info["images_dir_exists"]:
        image_pattern = str(images_dir / FILE_PATTERNS["images"])
        image_files = sorted(glob.glob(image_pattern))
        validation_info["image_files"] = image_files
    
    # Get mask files
    if validation_info["masks_dir_exists"]:
        mask_pattern = str(masks_dir / FILE_PATTERNS["masks"])
        mask_files = sorted(glob.glob(mask_pattern))
        validation_info["mask_files"] = mask_files
    
    # Check for matching pairs
    if validation_info["images_dir_exists"] and validation_info["masks_dir_exists"]:
        image_stems = {Path(f).stem for f in validation_info["image_files"]}
        mask_stems = {Path(f).stem for f in validation_info["mask_files"]}
        
        matching_stems = image_stems.intersection(mask_stems)
        validation_info["matching_pairs"] = len(matching_stems)
        
        # Check for mismatched files
        image_only = image_stems - mask_stems
        mask_only = mask_stems - image_stems
        
        if image_only:
            validation_info["errors"].append(f"Images without masks: {list(image_only)}")
        if mask_only:
            validation_info["errors"].append(f"Masks without images: {list(mask_only)}")
    
    # Determine if dataset is valid
    is_valid = (
        validation_info["images_dir_exists"] and
        validation_info["masks_dir_exists"] and
        validation_info["matching_pairs"] > 0 and
        len(validation_info["errors"]) == 0
    )
    
    return is_valid, validation_info


def create_image_list_file(output_file: str = "image_names.txt",
                          images_dir: Path = IMAGES_DIR) -> bool:
    """
    Create a text file with sorted image names.
    
    Args:
        output_file: Output file name
        images_dir: Directory containing images
        
    Returns:
        True if successful, False otherwise
    """
    try:
        if not images_dir.exists():
            logger.error(f"Images directory does not exist: {images_dir}")
            return False
        
        # Get image files
        image_pattern = str(images_dir / FILE_PATTERNS["images"])
        image_files = glob.glob(image_pattern)
        
        if not image_files:
            logger.error(f"No images found in {images_dir}")
            return False
        
        # Extract and sort file names
        image_names = []
        for file_path in image_files:
            file_name = Path(file_path).stem
            try:
                # Try to convert to integer for numerical sorting
                image_names.append((int(file_name), file_name))
            except ValueError:
                # If not numeric, use string sorting
                image_names.append((file_name, file_name))
        
        # Sort by the first element (numeric or string)
        image_names.sort(key=lambda x: x[0])
        
        # Write to file
        with open(output_file, "w") as f:
            for _, name in image_names:
                f.write(f"{name}\n")
        
        logger.info(f"Created image list file: {output_file} with {len(image_names)} images")
        return True
        
    except Exception as e:
        logger.error(f"Error creating image list file: {e}")
        return False


def get_dataset_info(images_dir: Path = IMAGES_DIR, 
                    masks_dir: Path = MASKS_DIR) -> dict:
    """
    Get information about the dataset.
    
    Args:
        images_dir: Directory containing images
        masks_dir: Directory containing masks
        
    Returns:
        Dictionary with dataset information
    """
    info = {
        "total_images": 0,
        "total_masks": 0,
        "matching_pairs": 0,
        "image_extensions": set(),
        "mask_extensions": set(),
        "image_sizes": [],
        "mask_sizes": []
    }
    
    # Get image files
    if images_dir.exists():
        image_pattern = str(images_dir / FILE_PATTERNS["images"])
        image_files = glob.glob(image_pattern)
        info["total_images"] = len(image_files)
        
        for file_path in image_files:
            ext = Path(file_path).suffix.lower()
            info["image_extensions"].add(ext)
    
    # Get mask files
    if masks_dir.exists():
        mask_pattern = str(masks_dir / FILE_PATTERNS["masks"])
        mask_files = glob.glob(mask_pattern)
        info["total_masks"] = len(mask_files)
        
        for file_path in mask_files:
            ext = Path(file_path).suffix.lower()
            info["mask_extensions"].add(ext)
    
    # Count matching pairs
    if images_dir.exists() and masks_dir.exists():
        image_stems = {Path(f).stem for f in glob.glob(str(images_dir / FILE_PATTERNS["images"]))}
        mask_stems = {Path(f).stem for f in glob.glob(str(masks_dir / FILE_PATTERNS["masks"]))}
        info["matching_pairs"] = len(image_stems.intersection(mask_stems))
    
    return info


def main():
    """
    Main function for data preprocessing.
    """
    logger.info("Starting data preprocessing...")
    
    # Validate dataset
    is_valid, validation_info = validate_dataset()
    
    if is_valid:
        logger.info("Dataset validation passed!")
        logger.info(f"Found {validation_info['matching_pairs']} image-mask pairs")
    else:
        logger.error("Dataset validation failed!")
        for error in validation_info["errors"]:
            logger.error(f"  - {error}")
        return
    
    # Get dataset info
    info = get_dataset_info()
    logger.info(f"Dataset info: {info}")
    
    # Create image list file
    if create_image_list_file():
        logger.info("Image list file created successfully")
    else:
        logger.error("Failed to create image list file")
    
    logger.info("Data preprocessing completed!")


if __name__ == "__main__":
    main() 