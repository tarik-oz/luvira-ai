"""
Special color handler for colors needing specific treatment.
"""

import numpy as np
import cv2

class SpecialColorHandler:
    """
    Handles special color transformations for specific colors 
    that need customized treatment.
    """
    
    def handle_blue_color(
        self, result_hsv: np.ndarray, image_hsv: np.ndarray, mask_normalized: np.ndarray
    ) -> np.ndarray:
        """
        Apply specialized transformation for blue hair.
        
        Args:
            result_hsv: Current HSV result to modify
            image_hsv: Original image in HSV
            mask_normalized: Hair mask (0-1 range)
            
        Returns:
            np.ndarray: Transformed HSV image for blue hair
        """
        # Convert RGB blue [0,0,255] to HSV to get the target hue
        blue_rgb = np.uint8([[[0, 0, 255]]])
        blue_hsv = cv2.cvtColor(blue_rgb, cv2.COLOR_RGB2HSV)
        target_hue = float(blue_hsv[0][0][0])  # Should be around 120 in OpenCV HSV
        
        # Get current hue and calculate shortest path to target
        current_hue = image_hsv[:,:,0]
        hue_diff = target_hue - current_hue
        
        # Ensure we take the shortest path around the hue circle (OpenCV HSV: 0-180)
        hue_diff = np.where(hue_diff > 90, hue_diff - 180, hue_diff)
        hue_diff = np.where(hue_diff < -90, hue_diff + 180, hue_diff)
        
        # Apply smooth hue transition
        new_hue = np.mod(current_hue + hue_diff * mask_normalized, 180)
        
        # Moderate saturation boost with lower minimum
        sat_boost = np.clip(image_hsv[:,:,1] * 1.3, 0, 255)  # Reduced from 1.6
        min_sat = 140  # Reduced from 180
        new_sat = np.where(sat_boost < min_sat, min_sat, sat_boost)
        
        # Brightness adjustment: boost shadows and mid-tones while preserving highlights
        val = image_hsv[:,:,2]
        new_val = np.where(
            val < 128,
            val * 1.4,  # Boost shadows more
            val * 1.2   # Boost mid-tones less
        )
        new_val = np.clip(new_val, 0, 255)
        
        # Combine channels with mask blending
        result_hsv[:,:,0] = new_hue * mask_normalized + result_hsv[:,:,0] * (1 - mask_normalized)
        result_hsv[:,:,1] = new_sat * mask_normalized + result_hsv[:,:,1] * (1 - mask_normalized)
        result_hsv[:,:,2] = new_val * mask_normalized + result_hsv[:,:,2] * (1 - mask_normalized)
        
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