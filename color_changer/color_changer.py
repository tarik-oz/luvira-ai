import cv2
import numpy as np

class HairColorChanger:
    """
    Advanced hair color changer with HSV color space for natural results.
    """

    @staticmethod
    def _preprocess_inputs(image, mask, target_color):
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

    @staticmethod
    def _analyze_hair_characteristics(image_float, mask_normalized, target_rgb):
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

    @staticmethod
    def _apply_hsv_transformations(image_hsv, mask_normalized, target_hsv, alpha, saturation_factor, brightness_adjustment):
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
        
        is_grey_target = target_hsv[1] < 60
        
        if is_grey_target:
            hair_pixels = (original_value * mask_normalized)[mask_normalized > 0.1]
            avg_hair_brightness = np.mean(hair_pixels) if len(hair_pixels) > 0 else 0
            
            sat_reduction = 0.9 if avg_hair_brightness < 50 else 0.95
            result_hsv[:,:,1] = np.where(mask_normalized > 0.1,
                                        np.clip(original_saturation * (1 - alpha * sat_reduction), 0, 255),
                                        original_saturation)
            
            target_value_factor = target_hsv[2] / 255.0
            if avg_hair_brightness < 50:
                value_boost = 1.0 + (target_value_factor - 0.1) * alpha * 1.2  # Daha güçlü boost
                result_hsv[:,:,2] = np.where(mask_normalized > 0.1,
                                            np.clip(original_value * value_boost, 0, 255),
                                            original_value)
            else:
                if target_value_factor < 0.3:
                    result_hsv[:,:,2] = np.where(mask_normalized > 0.1,
                                                np.clip(original_value * (0.3 + target_value_factor * 0.8), 0, 255),  # Daha güçlü
                                                original_value)
                elif target_value_factor > 0.7:
                    result_hsv[:,:,2] = np.where(mask_normalized > 0.1,
                                                np.clip(original_value * (0.7 + target_value_factor * 0.4), 0, 255),  # Daha güçlü
                                                original_value)
                else:
                    result_hsv[:,:,2] = np.where(mask_normalized > 0.1,
                                                np.clip(original_value * (0.5 + target_value_factor * 0.6), 0, 255),  # Daha güçlü
                                                original_value)
            
            result_hsv[:,:,0] = np.where(mask_normalized > 0.1,
                                        image_hsv[:,:,0] * (1 - alpha * 0.4),
                                        image_hsv[:,:,0])
        
            # ===== PRECISE COOL COLOR HANDLING =====
            hue = target_hsv[0]
            sat = target_hsv[1]
            
            # Identify blue and purple targets with precise hue ranges
            is_blue = (110 <= hue <= 125) and (sat > 150)
            is_purple = (145 <= hue <= 160) and (sat > 150)
            
            # ===== SPECIALIZED BLUE TRANSFORMATION =====
            if is_blue:
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
            
            # ===== SPECIALIZED PURPLE TRANSFORMATION =====
            elif is_purple:
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
    
    @staticmethod
    def _apply_natural_blending(image_float, result_rgb, mask_3ch, alpha):
        """
        Apply natural blending to preserve texture and lighting.
        
        Args:
            image_float: Original normalized image
            result_rgb: Color-changed normalized image
            mask_3ch: 3-channel mask
            alpha: Blending factor
        
        Returns:
            np.ndarray: Final blended image
        """
        # Use mask intensity to create smooth transitions
        smooth_mask = mask_3ch * alpha
        
        # Create luminance-based blending for more natural look
        original_luminance = 0.299 * image_float[:,:,0] + 0.587 * image_float[:,:,1] + 0.114 * image_float[:,:,2]
        target_luminance = 0.299 * result_rgb[:,:,0] + 0.587 * result_rgb[:,:,1] + 0.114 * result_rgb[:,:,2]
        
        # Blend based on luminance similarity - preserve natural texture
        luminance_diff = np.abs(original_luminance - target_luminance)
        luminance_factor = np.where(luminance_diff > 0.3, 0.8, 1.0)
        adaptive_blend = smooth_mask * np.stack([luminance_factor] * 3, axis=-1)
        
        # Final gentle blending
        result = image_float * (1 - adaptive_blend) + result_rgb * adaptive_blend
        
        return result

    @staticmethod
    def change_hair_color(image, mask, target_color):
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
            HairColorChanger._preprocess_inputs(image, mask, target_color)
        
        # Check if hair is detected
        if np.sum(mask_normalized > 0.1) == 0:
            return image_rgb  # No hair detected, return original
        
        # Analyze hair characteristics
        alpha, saturation_factor, brightness_adjustment = \
            HairColorChanger._analyze_hair_characteristics(image_float, mask_normalized, target_rgb)
        
        # Convert to HSV for better color control
        image_hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV).astype(np.float32)
        
        # Apply HSV transformations
        result_hsv = HairColorChanger._apply_hsv_transformations(
            image_hsv, mask_normalized, target_hsv, alpha, saturation_factor, brightness_adjustment
        )
        
        # Convert back to RGB
        result_rgb = cv2.cvtColor(result_hsv.astype(np.uint8), cv2.COLOR_HSV2RGB).astype(np.float32) / 255.0
        
        # Apply natural blending
        result = HairColorChanger._apply_natural_blending(image_float, result_rgb, mask_3ch, alpha)
        
        # Convert back to RGB and return
        result_rgb = np.clip(result * 255, 0, 255).astype(np.uint8)
        return result_rgb  # Return RGB format for web API