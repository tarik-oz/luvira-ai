import cv2
import numpy as np

class HairColorChanger:
    """
    Advanced hair color changer with HSV color space for natural results.
    """
    
    @staticmethod
    def change_hair_color(image, mask, rgb_color, alpha=0.5, saturation_factor=1.2):
        """
        Advanced hair color change with HSV color space.
        Preserves brightness and shadows while changing hue and saturation.
        
        Args:
            image: Original image (BGR or RGB, np.ndarray)
            mask: Grayscale mask (0-255, np.ndarray, single channel)
            rgb_color: Target hair color (e.g: [255, 0, 0] red)
            alpha: Color intensity (0-1)
            saturation_factor: Factor to adjust saturation (1.0 = no change)
        """
        if len(mask.shape) == 3:
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        
        # Normalize mask
        mask_normalized = mask.astype(np.float32) / 255.0
        
        # Convert to HSV for better color manipulation
        image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
        
        # Manually calculate HSV from RGB to avoid hue calculation issues
        r, g, b = rgb_color[0] / 255.0, rgb_color[1] / 255.0, rgb_color[2] / 255.0
        cmax = max(r, g, b)
        cmin = min(r, g, b)
        diff = cmax - cmin
        
        # Calculate hue
        if diff == 0:
            h = 0
        elif cmax == r:
            h = (60 * ((g - b) / diff) + 360) % 360
        elif cmax == g:
            h = (60 * ((b - r) / diff) + 120) % 360
        else:  # cmax == b
            h = (60 * ((r - g) / diff) + 240) % 360
        
        # Convert to OpenCV HSV range (0-180)
        h_cv = h / 2
        
        # Calculate saturation
        s = 0 if cmax == 0 else (diff / cmax) * 255
        
        # Value is the maximum component
        v = cmax * 255
        
        target_hsv = np.array([h_cv, s, v])
        
        # Create 3-channel mask
        mask_3ch = np.stack([mask_normalized] * 3, axis=-1)
        
        # Blend hue and saturation based on mask
        blended_hsv = image_hsv.copy()
        blended_hsv[:, :, 0] = image_hsv[:, :, 0] * (1 - mask_normalized) + target_hsv[0] * mask_normalized
        blended_hsv[:, :, 1] = (
            image_hsv[:, :, 1] * (1 - mask_normalized) +
            target_hsv[1] * saturation_factor * mask_normalized
        )
        
        # Preserve value (brightness) from original image
        # This maintains shadows and highlights
        blended_hsv[:, :, 2] = image_hsv[:, :, 2]
        
        # Convert back to BGR
        result_hsv = np.clip(blended_hsv, 0, 255).astype(np.uint8)
        result = cv2.cvtColor(result_hsv, cv2.COLOR_HSV2BGR)
        
        # Final blending with original
        final_result = image.astype(np.float32) * (1 - alpha * mask_3ch) + result.astype(np.float32) * alpha * mask_3ch
        
        return np.clip(final_result, 0, 255).astype(np.uint8)