"""
Base data loader class for hair segmentation dataset.
"""
import cv2
import numpy as np
import glob
from pathlib import Path
from typing import Tuple, List, Optional
from abc import ABC, abstractmethod

from config import IMAGES_DIR, MASKS_DIR, DATA_CONFIG, FILE_PATTERNS

class BaseDataLoader(ABC):
    def __init__(self, 
                 images_dir: Path = IMAGES_DIR,
                 masks_dir: Path = MASKS_DIR,
                 image_size: Tuple[int, int] = DATA_CONFIG["image_size"],
                 normalization_factor: float = DATA_CONFIG["normalization_factor"],
                 **kwargs):
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        self.image_size = image_size
        self.normalization_factor = normalization_factor

    def get_file_paths(self) -> Tuple[List[str], List[str]]:
        image_patterns = FILE_PATTERNS.get("images", ["*.jpg", "*.png"])
        mask_patterns = FILE_PATTERNS.get("masks", ["*.png", "*.webp"])

        image_paths = sorted([p for pattern in image_patterns for p in glob.glob(str(self.images_dir / pattern))])
        mask_paths = sorted([p for pattern in mask_patterns for p in glob.glob(str(self.masks_dir / pattern))])
        
        if not image_paths or not mask_paths:
            raise ValueError(f"No images or masks found in {self.images_dir} or {self.masks_dir}")
        
        print(f"Found {len(image_paths)} images and {len(mask_paths)} masks")
        return image_paths, mask_paths
        
    def _load_image(self, image_path: Path) -> Optional[np.ndarray]:
        try:
            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if image is None: raise ValueError("Image not loaded")
            image = cv2.resize(image, self.image_size)
            return (image / self.normalization_factor).astype(np.float32)
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            return None
    
    def _load_mask(self, mask_path: Path) -> Optional[np.ndarray]:
        try:
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if mask is None: raise ValueError("Mask not loaded")
            mask = cv2.resize(mask, self.image_size)
            return (mask / 255.0).astype(np.float32)
        except Exception as e:
            print(f"Error loading mask {mask_path}: {e}")
            return None
    
    def get_data_info(self) -> dict:
        """Get key information about the dataset configuration."""
        return {"image_size": self.image_size, "normalization_factor": self.normalization_factor}
    
    @abstractmethod
    def get_datasets(self, validation_split: float, random_seed: int):
        pass