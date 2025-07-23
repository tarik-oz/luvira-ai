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
        
        # Apply strong hue transition for blue
        new_hue = np.mod(current_hue + hue_diff * 0.9, 180)
        
        # High saturation boost for vibrant blue
        sat_boost = np.clip(image_hsv[:,:,1] * 1.5, 0, 255)
        min_sat = 180
        new_sat = np.where(sat_boost < min_sat, min_sat, sat_boost)
        
        # Brightness adjustment for blue depth
        val = image_hsv[:,:,2]
        new_val = np.where(
            val < 100,
            val * 1.3,  # Boost shadows more
            np.where(val > 180,
                    val * 0.9,  # Reduce highlights slightly
                    val * 1.1   # Boost mid-tones
            )
        )
        new_val = np.clip(new_val, 20, 255)
        
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
    
    def handle_grey_color(
        self, 
        result_hsv: np.ndarray, 
        image_hsv: np.ndarray, 
        mask_normalized: np.ndarray, 
        target_hsv: np.ndarray, 
        alpha: float
    ) -> np.ndarray:
        """
        Apply specialized transformations for grey/silver hair colors.
        
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
    
    def handle_auburn_color(
        self,
        result_hsv: np.ndarray,
        image_hsv: np.ndarray,
        mask_normalized: np.ndarray
    ) -> np.ndarray:
        """
        Apply specialized transformation for auburn hair.
        Auburn is a reddish-brown color that needs careful hue handling.
        
        Args:
            result_hsv: Current result HSV image
            image_hsv: Original HSV image
            mask_normalized: Normalized mask
            
        Returns:
            np.ndarray: Transformed HSV image for auburn hair
        """
        # Auburn target hue is around 10° (reddish-brown)
        auburn_hue = 10.0
        
        # Get current hue and apply transformation
        current_hue = image_hsv[:,:,0]
        
        # Calculate hue difference, handling wraparound
        hue_diff = auburn_hue - current_hue
        hue_diff = np.where(hue_diff > 90, hue_diff - 180, hue_diff)
        hue_diff = np.where(hue_diff < -90, hue_diff + 180, hue_diff)
        
        # Apply strong hue shift toward auburn
        result_hsv[:,:,0] = np.where(
            mask_normalized > 0.1,
            np.mod(current_hue + hue_diff * 0.85, 180),
            current_hue
        )
        
        # Boost saturation for rich auburn color
        current_sat = image_hsv[:,:,1]
        auburn_sat = np.clip(current_sat * 1.4, 100, 255)  # Minimum saturation of 100
        result_hsv[:,:,1] = np.where(
            mask_normalized > 0.1,
            auburn_sat,
            current_sat
        )
        
        # Adjust brightness for auburn warmth
        current_val = image_hsv[:,:,2]
        val_adjust = np.where(
            current_val < 80,
            current_val * 1.3,  # Boost dark areas
            np.where(current_val > 180,
                    current_val * 0.85,  # Tone down highlights
                    current_val * 1.1   # Slight boost for mid-tones
            )
        )
        result_hsv[:,:,2] = np.where(
            mask_normalized > 0.1,
            np.clip(val_adjust, 30, 220),
            current_val
        )
        
        return result_hsv
    
    def handle_copper_color(
        self,
        result_hsv: np.ndarray,
        image_hsv: np.ndarray,
        mask_normalized: np.ndarray
    ) -> np.ndarray:
        """
        Apply specialized transformation for copper hair.
        Copper is a warm orange-brown color.
        
        Args:
            result_hsv: Current result HSV image
            image_hsv: Original HSV image
            mask_normalized: Normalized mask
            
        Returns:
            np.ndarray: Transformed HSV image for copper hair
        """
        # Copper target hue is around 14° (orange-brown)
        copper_hue = 14.0
        
        # Get current hue and apply transformation
        current_hue = image_hsv[:,:,0]
        
        # Calculate hue difference, handling wraparound
        hue_diff = copper_hue - current_hue
        hue_diff = np.where(hue_diff > 90, hue_diff - 180, hue_diff)
        hue_diff = np.where(hue_diff < -90, hue_diff + 180, hue_diff)
        
        # Apply strong hue shift toward copper
        result_hsv[:,:,0] = np.where(
            mask_normalized > 0.1,
            np.mod(current_hue + hue_diff * 0.9, 180),
            current_hue
        )
        
        # High saturation for vibrant copper
        current_sat = image_hsv[:,:,1]
        copper_sat = np.clip(current_sat * 1.6, 150, 255)  # Minimum saturation of 150
        result_hsv[:,:,1] = np.where(
            mask_normalized > 0.1,
            copper_sat,
            current_sat
        )
        
        # Bright copper adjustment
        current_val = image_hsv[:,:,2]
        val_adjust = np.where(
            current_val < 100,
            current_val * 1.4,  # Boost shadows significantly
            np.where(current_val > 200,
                    current_val * 0.9,  # Slight reduction for highlights
                    current_val * 1.2   # Boost mid-tones
            )
        )
        result_hsv[:,:,2] = np.where(
            mask_normalized > 0.1,
            np.clip(val_adjust, 50, 240),
            current_val
        )
        
        return result_hsv
    
    def handle_pink_color(
        self,
        result_hsv: np.ndarray,
        image_hsv: np.ndarray,
        mask_normalized: np.ndarray
    ) -> np.ndarray:
        """
        Apply specialized transformation for pink hair.
        Pink requires vibrant saturation and specific hue handling.
        
        Args:
            result_hsv: Current result HSV image
            image_hsv: Original HSV image
            mask_normalized: Normalized mask
            
        Returns:
            np.ndarray: Transformed HSV image for pink hair
        """
        # Pink target hue is around 165° (magenta/pink)
        pink_hue = 165.0
        
        # Get current hue and apply transformation
        current_hue = image_hsv[:,:,0]
        
        # Calculate hue difference, handling wraparound
        hue_diff = pink_hue - current_hue
        hue_diff = np.where(hue_diff > 90, hue_diff - 180, hue_diff)
        hue_diff = np.where(hue_diff < -90, hue_diff + 180, hue_diff)
        
        # Apply strong hue shift toward pink
        result_hsv[:,:,0] = np.where(
            mask_normalized > 0.1,
            np.mod(current_hue + hue_diff * 0.9, 180),
            current_hue
        )
        
        # High saturation for vibrant pink
        current_sat = image_hsv[:,:,1]
        pink_sat = np.clip(current_sat * 1.8, 120, 255)  # Minimum saturation of 120
        result_hsv[:,:,1] = np.where(
            mask_normalized > 0.1,
            pink_sat,
            current_sat
        )
        
        # Brightness adjustment for pink vibrancy
        current_val = image_hsv[:,:,2]
        val_adjust = np.where(
            current_val < 80,
            current_val * 1.4,  # Boost dark areas significantly
            np.where(current_val > 200,
                    current_val * 0.95,  # Slight reduction for highlights
                    current_val * 1.15  # Boost mid-tones
            )
        )
        result_hsv[:,:,2] = np.where(
            mask_normalized > 0.1,
            np.clip(val_adjust, 40, 230),
            current_val
        )
        
        return result_hsv 