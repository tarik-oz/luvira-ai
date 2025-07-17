"""
Main color transformer class that orchestrates the hair color transformation process.
"""

import cv2
import numpy as np
from typing import List, Tuple

from color_changer.transformers.hsv_transformer import HsvTransformer
from color_changer.transformers.blender import Blender


class ColorTransformer:
    """
    Main class for hair color transformation, orchestrates the process by delegating
    to specialized components for different transformation steps.
    """
    
    def __init__(self):
        self.hsv_transformer = HsvTransformer()
        self.blender = Blender()
    
    def change_hair_color(self, image: np.ndarray, mask: np.ndarray, target_color: List[int]) -> np.ndarray:
        """
        Natural hair recoloring that preserves original texture and lighting.
        Focuses on subtle, realistic color changes.
        
        Args:
            image: Original image (BGR format, np.ndarray)
            mask: Grayscale mask (0-255, np.ndarray) 
            target_color: Target hair color [R, G, B] (0-255)
        
        Returns:
            Recolored image (RGB format, np.ndarray)
        """
        # Preprocess inputs
        image_rgb, image_float, mask_normalized, mask_3ch, target_rgb, target_hsv = \
            self._preprocess_inputs(image, mask, target_color)
        
        # Check if hair is detected
        if np.sum(mask_normalized > 0.1) == 0:
            return image_rgb  # No hair detected, return original
        
        # Analyze hair characteristics
        alpha, saturation_factor, brightness_adjustment = \
            self._analyze_hair_characteristics(image_float, mask_normalized, target_rgb)
        
        # Convert to HSV for better color control
        image_hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV).astype(np.float32)
        
        # Apply HSV transformations
        result_hsv = self.hsv_transformer.apply_hsv_transformations(
            image_hsv, mask_normalized, target_hsv, alpha, saturation_factor, brightness_adjustment
        )
        
        # Convert back to RGB
        result_rgb = cv2.cvtColor(result_hsv.astype(np.uint8), cv2.COLOR_HSV2RGB).astype(np.float32) / 255.0
        
        # Apply natural blending
        result = self.blender.apply_natural_blending(image_float, result_rgb, mask_3ch, alpha)
        
        # Convert back to RGB and return
        result_rgb = np.clip(result * 255, 0, 255).astype(np.uint8)
        return result_rgb  # Return RGB format for web API
    
    def _preprocess_inputs(self, image: np.ndarray, mask: np.ndarray, target_color: List[int]) -> Tuple:
        """
        Preprocess and validate input images and mask.
        
        Args:
            image: Original image (BGR format, np.ndarray)
            mask: Grayscale mask (0-255, np.ndarray) 
            target_color: Target hair color [R, G, B] (0-255)
        
        Returns:
            tuple: (image_rgb, image_float, mask_normalized, mask_3ch, target_rgb, target_hsv)
        """
        # Input validation and preprocessing
        if len(mask.shape) == 3:
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        
        mask_normalized = mask.astype(np.float32) / 255.0
        mask_3ch = np.stack([mask_normalized] * 3, axis=-1)
        
        # Convert BGR to RGB for proper color handling
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_float = image_rgb.astype(np.float32) / 255.0
        target_rgb = np.array(target_color, dtype=np.float32) / 255.0
        
        # Convert target color to HSV
        target_hsv = cv2.cvtColor(np.uint8([[target_color]]), cv2.COLOR_RGB2HSV)[0][0].astype(np.float32)
        
        return image_rgb, image_float, mask_normalized, mask_3ch, target_rgb, target_hsv
    
    def _analyze_hair_characteristics(self, image_float: np.ndarray, mask_normalized: np.ndarray, target_rgb: np.ndarray) -> Tuple[float, float, float]:
        """
        Analyze hair characteristics to determine color change parameters.
        
        Args:
            image_float: Normalized RGB image (0-1)
            mask_normalized: Normalized mask (0-1)
            target_rgb: Target color in RGB (0-1)
        
        Returns:
            tuple: (alpha, saturation_factor, brightness_adjustment)
        """
        hair_pixels = image_float[mask_normalized > 0.1]
        if len(hair_pixels) == 0:
            return 0.0, 1.0, 0.0  # No hair detected
        
        hair_brightness = np.mean(hair_pixels)
        target_brightness = np.mean(target_rgb)
        brightness_diff = abs(hair_brightness - target_brightness)

        # Continuous parameter adjustment based on brightness difference
        max_diff = 0.4  # Lowered for stronger effect even on smaller differences
        norm_diff = min(1.0, brightness_diff / max_diff)

        alpha = 0.8 + 0.15 * norm_diff
        saturation_factor = 1.2 + 0.5 * norm_diff
        brightness_adjustment = 0.3 + 0.4 * norm_diff

        return alpha, saturation_factor, brightness_adjustment 