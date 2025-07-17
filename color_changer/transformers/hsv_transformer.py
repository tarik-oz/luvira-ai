"""
HSV transformer for hair color changing operations.
"""

import numpy as np
from typing import Tuple

from color_changer.transformers.special_color_handler import SpecialColorHandler

class HsvTransformer:
    """
    Handles HSV color space transformations for hair color changing.
    """
    
    def __init__(self):
        self.special_color_handler = SpecialColorHandler()
    
    def apply_hsv_transformations(
        self, 
        image_hsv: np.ndarray, 
        mask_normalized: np.ndarray, 
        target_hsv: np.ndarray, 
        alpha: float, 
        saturation_factor: float, 
        brightness_adjustment: float
    ) -> np.ndarray:
        """
        Apply HSV transformations for color change.
        
        Args:
            image_hsv: Image in HSV format
            mask_normalized: Normalized mask (0-1)
            target_hsv: Target color in HSV
            alpha: Blending factor
            saturation_factor: Saturation boost factor
            brightness_adjustment: Brightness adjustment factor
        
        Returns:
            np.ndarray: Transformed HSV image
        """
        result_hsv = image_hsv.copy()
        
        # HUE: Softer blend towards target hue
        hue_diff = target_hsv[0] - image_hsv[:,:,0]
        hue_diff = np.where(hue_diff > 90, hue_diff - 180, hue_diff)
        hue_diff = np.where(hue_diff < -90, hue_diff + 180, hue_diff)
        
        result_hsv[:,:,0] = np.where(mask_normalized > 0.1, 
                                     image_hsv[:,:,0] + hue_diff * (alpha * 0.95),
                                     image_hsv[:,:,0])
        result_hsv[:,:,0] = np.clip(result_hsv[:,:,0] % 180, 0, 179)
        
        # SATURATION: Restore conditional boost, with extra for fantasy
        original_saturation = image_hsv[:,:,1]
        saturation_boost = np.where(original_saturation > 200, 1.0, saturation_factor)
        result_hsv[:,:,1] = np.where(mask_normalized > 0.1,
                                     np.clip(original_saturation * saturation_boost, 0, 255),
                                     original_saturation)
        if target_hsv[1] > 200:  # Fantasy high-sat colors
            result_hsv[:,:,1] = np.where(mask_normalized > 0.1,
                                         np.clip(result_hsv[:,:,1] + (target_hsv[1] - result_hsv[:,:,1]) * alpha * 0.5, 0, 255),
                                         result_hsv[:,:,1])
        
        # VALUE: Restore conditional brighten/darken for natural preservation
        original_value = image_hsv[:,:,2]
        hair_pixels = (original_value * mask_normalized)[mask_normalized > 0.1]
        if len(hair_pixels) > 0:
            hair_brightness = np.mean(hair_pixels) / 255.0
            target_brightness = target_hsv[2] / 255.0
            
            if hair_brightness < target_brightness:
                brightness_boost = 1.0 + (brightness_adjustment * (target_brightness - hair_brightness) / target_brightness)
                brightness_boost = np.where(original_value > 180, 1.0, brightness_boost)
                result_hsv[:,:,2] = np.where(mask_normalized > 0.1,
                                             np.clip(original_value * brightness_boost, 0, 255),
                                             original_value)
            elif hair_brightness > target_brightness:
                darkness_factor = 1.0 - (brightness_adjustment * (hair_brightness - target_brightness) / hair_brightness)
                darkness_factor = np.where(original_value < 50, 1.0, darkness_factor)
                result_hsv[:,:,2] = np.where(mask_normalized > 0.1,
                                             np.clip(original_value * darkness_factor, 0, 255),
                                             original_value)
        
        # Check for special colors like gray, blue or purple
        is_grey_target = target_hsv[1] < 60
        
        # Apply special color transformations if needed
        if is_grey_target:
            result_hsv = self._apply_grey_transformations(
                result_hsv, image_hsv, mask_normalized, target_hsv, alpha
            )
        else:
            # ===== PRECISE COOL COLOR HANDLING =====
            hue = target_hsv[0]
            sat = target_hsv[1]
            
            # Identify blue and purple targets with precise hue ranges
            is_blue = (110 <= hue <= 125) and (sat > 150)
            is_purple = (145 <= hue <= 160) and (sat > 150)
            
            # ===== SPECIALIZED COLOR TRANSFORMATIONS =====
            if is_blue:
                result_hsv = self.special_color_handler.handle_blue_color(
                    result_hsv, image_hsv, mask_normalized
                )
            elif is_purple:
                result_hsv = self.special_color_handler.handle_purple_color(
                    result_hsv, image_hsv, mask_normalized
                )
            
        return result_hsv
    
    def _apply_grey_transformations(
        self, 
        result_hsv: np.ndarray, 
        image_hsv: np.ndarray, 
        mask_normalized: np.ndarray, 
        target_hsv: np.ndarray, 
        alpha: float
    ) -> np.ndarray:
        """
        Apply transformations specific to grey/silver hair colors.
        
        Args:
            result_hsv: Current result in HSV
            image_hsv: Original image in HSV
            mask_normalized: Normalized mask
            target_hsv: Target color in HSV
            alpha: Blending factor
            
        Returns:
            np.ndarray: Transformed HSV image for grey color
        """
        original_saturation = image_hsv[:,:,1]
        original_value = image_hsv[:,:,2]
        
        hair_pixels = (original_value * mask_normalized)[mask_normalized > 0.1]
        avg_hair_brightness = np.mean(hair_pixels) if len(hair_pixels) > 0 else 0
        
        # Reduce saturation for grey colors
        sat_reduction = 0.9 if avg_hair_brightness < 50 else 0.95
        result_hsv[:,:,1] = np.where(mask_normalized > 0.1,
                                    np.clip(original_saturation * (1 - alpha * sat_reduction), 0, 255),
                                    original_saturation)
        
        # Adjust value/brightness based on target
        target_value_factor = target_hsv[2] / 255.0
        if avg_hair_brightness < 50:
            value_boost = 1.0 + (target_value_factor - 0.1) * alpha * 1.2
            result_hsv[:,:,2] = np.where(mask_normalized > 0.1,
                                        np.clip(original_value * value_boost, 0, 255),
                                        original_value)
        else:
            if target_value_factor < 0.3:
                result_hsv[:,:,2] = np.where(mask_normalized > 0.1,
                                            np.clip(original_value * (0.3 + target_value_factor * 0.8), 0, 255),
                                            original_value)
            elif target_value_factor > 0.7:
                result_hsv[:,:,2] = np.where(mask_normalized > 0.1,
                                            np.clip(original_value * (0.7 + target_value_factor * 0.4), 0, 255),
                                            original_value)
            else:
                result_hsv[:,:,2] = np.where(mask_normalized > 0.1,
                                            np.clip(original_value * (0.5 + target_value_factor * 0.6), 0, 255),
                                            original_value)
        
        # Reduce hue influence for grey
        result_hsv[:,:,0] = np.where(mask_normalized > 0.1,
                                    image_hsv[:,:,0] * (1 - alpha * 0.4),
                                    image_hsv[:,:,0])
        
        return result_hsv 