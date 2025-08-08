"""
HSV transformer for hair color changing operations.
"""

import numpy as np

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
        
        # Apply special color transformations based on TARGET color
        hue = target_hsv[0]
        sat = target_hsv[1]
        is_gray = sat < 60
        is_blue = (110 <= hue <= 125) and (sat > 150)
        is_purple = (145 <= hue <= 160) and (sat > 150)
        is_auburn = (5 <= hue <= 12) and (sat > 100)  # Auburn range
        is_copper = (12 <= hue <= 18) and (sat > 120)  # Copper orange range (H=14)
        is_pink = (160 <= hue <= 170) and (sat > 100)  # Pink/magenta range

        if is_gray:
            result_hsv = self.special_color_handler.handle_gray_color(
                result_hsv, image_hsv, mask_normalized, target_hsv, alpha
            )
        elif is_blue:
            result_hsv = self.special_color_handler.handle_blue_color(
                result_hsv, image_hsv, mask_normalized, target_hsv, alpha
            )
        elif is_purple:
            result_hsv = self.special_color_handler.handle_purple_color(
                result_hsv, image_hsv, mask_normalized, target_hsv, alpha
            )
        elif is_auburn:
            result_hsv = self.special_color_handler.handle_auburn_color(
                result_hsv, image_hsv, mask_normalized, target_hsv, alpha
            )
        elif is_copper:
            result_hsv = self.special_color_handler.handle_copper_color(
                result_hsv, image_hsv, mask_normalized, target_hsv, alpha
            )
        elif is_pink:
            result_hsv = self.special_color_handler.handle_pink_color(
                result_hsv, image_hsv, mask_normalized, target_hsv, alpha
            )
        # If gray hair + colored target, default processing (no special gray handling)
            
        return result_hsv
