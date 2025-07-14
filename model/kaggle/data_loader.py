"""
Kaggle-specific data loader for hair segmentation dataset.
Uses Kaggle paths and configuration for dataset loading.
Enhanced with data augmentation.
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

try:
    from .config import (
        IMAGES_DIR, MASKS_DIR, PROCESSED_DATA_DIR, 
        DATA_CONFIG, TRAINING_CONFIG, FILE_PATTERNS
    )
    from ..data_loader.base_data_loader import BaseDataLoader
    from ..data_loader.lazy_data_loader import LazyDataLoader
    from ..data_loader.traditional_data_loader import TraditionalDataLoader
    from .lazy_dataset import KaggleLazyDataset
except ImportError:
    # Fallback for direct execution
    from config import (
        IMAGES_DIR, MASKS_DIR, PROCESSED_DATA_DIR, 
        DATA_CONFIG, TRAINING_CONFIG, FILE_PATTERNS
    )
    from model.data_loader.base_data_loader import BaseDataLoader
    from model.data_loader.lazy_data_loader import LazyDataLoader
    from model.data_loader.traditional_data_loader import TraditionalDataLoader
    from lazy_dataset import KaggleLazyDataset

import logging

logger = logging.getLogger(__name__)

class KaggleDataLoader(LazyDataLoader):
    """
    Kaggle-specific data loader for hair segmentation dataset.
    Inherits from LazyDataLoader for memory efficiency with large datasets.
    Enhanced with data augmentation capabilities.
    """
    
    def __init__(self, **kwargs):
        """
        Initialize Kaggle data loader.
        
        Args:
            **kwargs: Additional configuration parameters
        """
        # Override paths with Kaggle-specific paths
        kaggle_config = {
            'images_dir': IMAGES_DIR,
            'masks_dir': MASKS_DIR,
            'processed_dir': PROCESSED_DATA_DIR,
            'image_size': DATA_CONFIG["image_size"],
            'normalization_factor': DATA_CONFIG["normalization_factor"]
        }
        
        # Update with any provided kwargs
        kaggle_config.update(kwargs)
        
        # Call parent constructor
        super().__init__(**kaggle_config)
        
        # Configure augmentation
        self.enable_augmentations = kwargs.get('apply_augmentations', True)
        
        logger.info("Kaggle data loader initialized")
        logger.info(f"Images directory: {self.images_dir}")
        logger.info(f"Masks directory: {self.masks_dir}")
        logger.info(f"Processed directory: {self.processed_dir}")
        logger.info(f"Data augmentation enabled: {self.enable_augmentations}")
        
        # Verify Kaggle dataset structure
        self._verify_kaggle_dataset()
    
    def _verify_kaggle_dataset(self):
        """
        Verify that the Kaggle dataset has the expected structure.
        """
        logger.info("Verifying Kaggle dataset structure...")
        
        # Check if directories exist
        if not self.images_dir.exists():
            raise FileNotFoundError(f"Images directory not found: {self.images_dir}")
        
        if not self.masks_dir.exists():
            raise FileNotFoundError(f"Masks directory not found: {self.masks_dir}")
        
        # Check for files
        image_files = []
        for pattern in FILE_PATTERNS["images"]:
            image_files.extend(list(self.images_dir.glob(pattern)))
        
        mask_files = []
        for pattern in FILE_PATTERNS["masks"]:
            mask_files.extend(list(self.masks_dir.glob(pattern)))
        
        logger.info(f"Found {len(image_files)} images and {len(mask_files)} masks")
        
        if len(image_files) == 0:
            raise ValueError("No image files found in the dataset")
        
        if len(mask_files) == 0:
            raise ValueError("No mask files found in the dataset")
        
        if len(image_files) != len(mask_files):
            logger.warning(f"Image count ({len(image_files)}) != Mask count ({len(mask_files)})")
            logger.warning("This may cause issues during training")
        
        # Log sample files
        logger.info("Sample image files:")
        for i, img_file in enumerate(image_files[:3]):
            logger.info(f"  {i+1}. {img_file.name}")
        
        logger.info("Sample mask files:")
        for i, mask_file in enumerate(mask_files[:3]):
            logger.info(f"  {i+1}. {mask_file.name}")
        
        expected_images = 30000  # Expected 30K dataset size
        if len(image_files) < expected_images * 0.25:  # Allow some flexibility
            logger.info(f"Found {len(image_files)} images, which is smaller than expected 30K dataset.")
            logger.info("This might be a sample dataset. Results may not be optimal for a small dataset.")
        else:
            logger.info(f"Dataset size looks good: {len(image_files)} images")
        
        logger.info("Kaggle dataset structure verified successfully")
    
    def create_datasets(self, validation_split: float = TRAINING_CONFIG["validation_split"],
                       random_seed: int = TRAINING_CONFIG["random_seed"]):
        """
        Create train and validation datasets with augmentation for training.
        
        Args:
            validation_split: Fraction of data to use for validation
            random_seed: Random seed for reproducibility
            
        Returns:
            Tuple of (train_dataset, val_dataset)
        """
        # Split paths
        train_img_paths, train_mask_paths, val_img_paths, val_mask_paths = self.split_data(
            validation_split, random_seed
        )
        
        # Create datasets with augmentation for training only
        train_dataset = KaggleLazyDataset(
            train_img_paths, 
            train_mask_paths,
            self.image_size, 
            self.normalization_factor,
            apply_augmentations=self.enable_augmentations,
            is_training=True
        )
        
        val_dataset = KaggleLazyDataset(
            val_img_paths, 
            val_mask_paths,
            self.image_size, 
            self.normalization_factor,
            apply_augmentations=False,  # No augmentation for validation
            is_training=False
        )
        
        logger.info(f"Created datasets with augmentation - Train: {len(train_dataset)}, Val: {len(val_dataset)}")
        
        return train_dataset, val_dataset
    
    def get_data_info(self) -> dict:
        """
        Get information about the Kaggle dataset.
        
        Returns:
            Dictionary with dataset information
        """
        base_info = super().get_data_info()
        
        # Add Kaggle-specific information
        kaggle_info = {
            "platform": "Kaggle",
            "dataset_location": "Kaggle Input",
            "images_path": str(self.images_dir),
            "masks_path": str(self.masks_dir),
            "processed_path": str(self.processed_dir),
            "lazy_loading": True,
            "memory_efficient": True,
            "augmentations_applied": self.enable_augmentations,
            "augmentations": [
                "HorizontalFlip",
                "RandomRotation (±10°)",
                "ColorJitter",
                "RandomBrightnessContrast",
                "ElasticTransform (mild)"
            ] if self.enable_augmentations else []
        }
        
        # Merge with base info
        base_info.update(kaggle_info)
        
        return base_info


def create_kaggle_data_loader(**kwargs) -> KaggleDataLoader:
    """
    Create a Kaggle data loader instance with augmentation.
    
    Args:
        **kwargs: Configuration parameters
        
    Returns:
        KaggleDataLoader instance
    """
    return KaggleDataLoader(**kwargs)


# Factory function for compatibility
def create_kaggle_lazy_data_loader(**kwargs) -> KaggleDataLoader:
    """
    Create a Kaggle lazy data loader (alias for create_kaggle_data_loader).
    
    Args:
        **kwargs: Configuration parameters
        
    Returns:
        KaggleDataLoader instance
    """
    return create_kaggle_data_loader(**kwargs) 