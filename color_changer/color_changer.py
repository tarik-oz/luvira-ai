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
        
        # Determine change intensity based on brightness difference
        is_major_change = brightness_diff > 0.3
        is_moderate_change = brightness_diff > 0.15
        
        if is_major_change:
            alpha = 0.9
            saturation_factor = 1.4
            brightness_adjustment = 0.6
        elif is_moderate_change:
            alpha = 0.8
            saturation_factor = 1.2
            brightness_adjustment = 0.4
        else:
            alpha = 0.7
            saturation_factor = 1.1
            brightness_adjustment = 0.2
        
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
        
        # HUE: Gentle transition
        hue_diff = target_hsv[0] - image_hsv[:,:,0]
        # Handle hue wrapping (0-179 in OpenCV)
        hue_diff = np.where(hue_diff > 90, hue_diff - 180, hue_diff)
        hue_diff = np.where(hue_diff < -90, hue_diff + 180, hue_diff)
        
        result_hsv[:,:,0] = np.where(mask_normalized > 0.1, 
                                    image_hsv[:,:,0] + hue_diff * alpha,
                                    image_hsv[:,:,0])
        result_hsv[:,:,0] = np.clip(result_hsv[:,:,0], 0, 179)
        
        # SATURATION: Gentle boost
        original_saturation = image_hsv[:,:,1]
        saturation_boost = np.where(original_saturation > 200, 1.0, saturation_factor)
        result_hsv[:,:,1] = np.where(mask_normalized > 0.1,
                                    np.clip(original_saturation * saturation_boost, 0, 255),
                                    original_saturation)
        
        # VALUE (BRIGHTNESS): Preserve natural lighting
        original_value = image_hsv[:,:,2]
        
        # Calculate hair brightness for comparison
        hair_pixels = (image_hsv[:,:,2] * mask_normalized)[mask_normalized > 0.1]
        if len(hair_pixels) > 0:
            hair_brightness = np.mean(hair_pixels) / 255.0
            target_brightness = target_hsv[2] / 255.0
            
            if hair_brightness < target_brightness:
                # Brighten
                brightness_boost = 1.0 + (brightness_adjustment * (target_brightness - hair_brightness) / target_brightness)
                brightness_boost = np.where(original_value > 180, 1.0, brightness_boost)
                result_hsv[:,:,2] = np.where(mask_normalized > 0.1,
                                           np.clip(original_value * brightness_boost, 0, 255),
                                           original_value)
            elif hair_brightness > target_brightness:
                # Darken
                darkness_factor = 1.0 - (brightness_adjustment * (hair_brightness - target_brightness) / hair_brightness)
                darkness_factor = np.where(original_value < 50, 1.0, darkness_factor)
                result_hsv[:,:,2] = np.where(mask_normalized > 0.1,
                                           np.clip(original_value * darkness_factor, 0, 255),
                                           original_value)
        
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