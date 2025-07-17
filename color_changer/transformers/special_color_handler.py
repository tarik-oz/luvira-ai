"""
Special color handler for colors needing specific treatment.
"""

import numpy as np

class SpecialColorHandler:
    """
    Handles special color transformations for specific colors 
    that need customized treatment.
    """
    
    def handle_blue_color(
        self,
        result_hsv: np.ndarray,
        image_hsv: np.ndarray,
        mask_normalized: np.ndarray
    ) -> np.ndarray:
        """
        Apply specialized transformation for blue hair.
        
        Args:
            result_hsv: Current result HSV image
            image_hsv: Original HSV image
            mask_normalized: Normalized mask
            
        Returns:
            np.ndarray: Transformed HSV image for blue hair
        """
        # Shift hue toward true blue (120°)
        blue_hue = 120.0
        
        # Apply hue transformation with anti-purple bias
        hue_diff = blue_hue - image_hsv[:,:,0]
        # Adjust large differences to avoid wrapping issues
        hue_diff = np.where(hue_diff > 90, hue_diff - 180, hue_diff)
        hue_diff = np.where(hue_diff < -90, hue_diff + 180, hue_diff)
        
        result_hsv[:,:,0] = np.where(
            mask_normalized > 0.1,
            image_hsv[:,:,0] + hue_diff * 0.95,  # Strong shift to true blue
            image_hsv[:,:,0]
        )
        
        # Saturation boost with blue-specific curve
        blue_sat_curve = np.clip(image_hsv[:,:,1] * 1.6, 0, 255)
        blue_sat_curve = np.where(blue_sat_curve < 180, 180, blue_sat_curve)  # Minimum saturation
        result_hsv[:,:,1] = np.where(
            mask_normalized > 0.1,
            blue_sat_curve,
            image_hsv[:,:,1]
        )
        
        # Brightness mapping for blue vibrancy
        current_val = result_hsv[:,:,2]
        # Boost mid-tones for maximum blue impact
        val_adjust = np.clip((current_val - 100) * 0.4, 0, 80)
        # Preserve highlights
        val_adjust = np.where(current_val > 180, 0, val_adjust)
        result_hsv[:,:,2] = np.where(
            mask_normalized > 0.1,
            np.clip(current_val + val_adjust, 0, 220),
            current_val
        )
        
        # Anti-purple correction
        purple_mask = (result_hsv[:,:,0] > 130) & (mask_normalized > 0.1)
        result_hsv[:,:,0] = np.where(
            purple_mask,
            np.clip(result_hsv[:,:,0] - 15, 100, 130),  # Shift away from purple
            result_hsv[:,:,0]
        )
        
        return result_hsv
    
    def handle_purple_color(
        self,
        result_hsv: np.ndarray,
        image_hsv: np.ndarray,
        mask_normalized: np.ndarray
    ) -> np.ndarray:
        """
        Apply specialized transformation for purple hair.
        
        Args:
            result_hsv: Current result HSV image
            image_hsv: Original HSV image
            mask_normalized: Normalized mask
            
        Returns:
            np.ndarray: Transformed HSV image for purple hair
        """
        # Target true purple (150°)
        purple_hue = 150.0
        
        # Apply hue transformation with anti-pink bias
        hue_diff = purple_hue - image_hsv[:,:,0]
        # Adjust large differences to avoid wrapping issues
        hue_diff = np.where(hue_diff > 90, hue_diff - 180, hue_diff)
        hue_diff = np.where(hue_diff < -90, hue_diff + 180, hue_diff)
        
        result_hsv[:,:,0] = np.where(
            mask_normalized > 0.1,
            image_hsv[:,:,0] + hue_diff * 0.92,  # Strong shift to true purple
            image_hsv[:,:,0]
        )
        
        # Saturation boost with purple-specific curve
        purple_sat_curve = np.clip(image_hsv[:,:,1] * 1.8, 0, 255)
        purple_sat_curve = np.where(purple_sat_curve < 190, 190, purple_sat_curve)  # Minimum saturation
        result_hsv[:,:,1] = np.where(
            mask_normalized > 0.1,
            purple_sat_curve,
            image_hsv[:,:,1]
        )
        
        # Brightness mapping for rich purple
        current_val = result_hsv[:,:,2]
        # Balanced adjustment curve
        val_adjust = np.where(
            current_val < 100,
            (100 - current_val) * 0.5,  # Boost shadows
            np.where(current_val > 180,
                    (current_val - 180) * -0.4,  # Reduce highlights
                    0  # Leave mid-tones
            )
        )
        result_hsv[:,:,2] = np.where(
            mask_normalized > 0.1,
            np.clip(current_val + val_adjust, 50, 200),
            current_val
        )
        
        # Anti-pink correction
        pink_mask = (result_hsv[:,:,0] < 140) & (mask_normalized > 0.1)
        result_hsv[:,:,0] = np.where(
            pink_mask,
            np.clip(result_hsv[:,:,0] + 10, 140, 160),  # Shift away from pink
            result_hsv[:,:,0]
        )
        
        return result_hsv 