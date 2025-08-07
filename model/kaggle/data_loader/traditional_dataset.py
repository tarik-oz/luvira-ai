"""
PyTorch Dataset for pre-loaded images and masks.
"""
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Tuple

from utils.augmentation import Augmentation

class TraditionalDataset(Dataset):
    def __init__(self, 
                 images: np.ndarray, 
                 masks: np.ndarray, 
                 use_augmentation: bool = False):
        self.images = images
        self.masks = masks
        self.use_augmentation = use_augmentation
        
    def __len__(self) -> int:
        return len(self.images)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        image, mask = self.images[idx].copy(), self.masks[idx].copy()
        
        if self.use_augmentation:
            image, mask = Augmentation.apply_augmentations(image, mask)
        
        image_tensor = torch.from_numpy(image).permute(2, 0, 1)
        mask_tensor = torch.from_numpy(mask).unsqueeze(0)
        
        return image_tensor, mask_tensor
