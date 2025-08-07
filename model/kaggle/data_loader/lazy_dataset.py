"""
PyTorch Dataset for on-demand loading of images and masks.
"""
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import List, Tuple

from utils.augmentation import Augmentation

class LazyDataset(Dataset):
    def __init__(self, 
                 image_paths: List[str], 
                 mask_paths: List[str], 
                 image_size: Tuple[int, int],
                 normalization_factor: float,
                 use_augmentation: bool = False):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.image_size = image_size
        self.normalization_factor = normalization_factor
        self.use_augmentation = use_augmentation
        
    def __len__(self) -> int:
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        image = cv2.imread(str(self.image_paths[idx]), cv2.IMREAD_COLOR)
        mask = cv2.imread(str(self.mask_paths[idx]), cv2.IMREAD_GRAYSCALE)
        
        if image is None or mask is None:
            raise ValueError(f"Could not load data at index {idx}")
        
        if self.use_augmentation:
            image, mask = Augmentation.apply_augmentations(image, mask)
        
        image = cv2.resize(image, self.image_size)
        mask = cv2.resize(mask, self.image_size)
        
        image = (image.astype(np.float32) / self.normalization_factor)
        mask = (mask.astype(np.float32) / 255.0)
        
        image_tensor = torch.from_numpy(image).permute(2, 0, 1)
        mask_tensor = torch.from_numpy(mask).unsqueeze(0)
        
        return image_tensor, mask_tensor
