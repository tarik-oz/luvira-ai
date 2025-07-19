"""
Color utilities for hair color change operations.
"""

import cv2
import numpy as np
from typing import List, Tuple, Dict

class ColorUtils:
    """
    Utility functions for color operations.
    """
    
    @staticmethod
    def rgb_to_hsv(rgb: List[int]) -> List[int]:
        """
        Convert RGB color to HSV.
        
        Args:
            rgb: RGB color [R, G, B] (0-255)
            
        Returns:
            HSV color [H, S, V] (H: 0-179, S/V: 0-255)
        """
        rgb_arr = np.array([[rgb]], dtype=np.uint8)
        hsv = cv2.cvtColor(rgb_arr, cv2.COLOR_RGB2HSV)[0][0]
        return hsv.tolist()
    
    @staticmethod
    def hsv_to_rgb(hsv: List[int]) -> List[int]:
        """
        Convert HSV color to RGB.
        
        Args:
            hsv: HSV color [H, S, V] (H: 0-179, S/V: 0-255)
            
        Returns:
            RGB color [R, G, B] (0-255)
        """
        hsv_arr = np.array([[hsv]], dtype=np.uint8)
        rgb = cv2.cvtColor(hsv_arr, cv2.COLOR_HSV2RGB)[0][0]
        return rgb.tolist()
    
    @staticmethod
    def generate_tonal_variations(
        base_rgb: List[int], 
        tone_configs: Dict[str, Dict] = None
    ) -> Dict[str, List[int]]:
        """
        Generate tonal variations of a base color using HSV adjustments.
        
        Args:
            base_rgb: Base RGB color [R, G, B] (0-255)
            tone_configs: Dictionary of tone configurations with saturation and brightness factors
                         Format: {"tone_name": {"saturation_factor": float, "brightness_factor": float}}
        
        Returns:
            Dictionary mapping tone names to RGB colors
        """
        if tone_configs is None:
            # Default tone configurations
            tone_configs = {
                "light": {"saturation_factor": 0.7, "brightness_factor": 1.3},
                "natural": {"saturation_factor": 1.0, "brightness_factor": 1.0},
                "vibrant": {"saturation_factor": 1.4, "brightness_factor": 1.1},
                "deep": {"saturation_factor": 1.2, "brightness_factor": 0.8},
                "muted": {"saturation_factor": 0.5, "brightness_factor": 0.9}
            }
        
        # Convert base color to HSV
        base_hsv = ColorUtils.rgb_to_hsv(base_rgb)
        h, s, v = base_hsv
        
        tonal_variations = {}
        
        for tone_name, config in tone_configs.items():
            # Apply saturation and brightness adjustments
            new_s = int(np.clip(s * config["saturation_factor"], 0, 255))
            new_v = int(np.clip(v * config["brightness_factor"], 0, 255))
            
            # Keep the same hue, adjust saturation and value
            new_hsv = [h, new_s, new_v]
            new_rgb = ColorUtils.hsv_to_rgb(new_hsv)
            
            tonal_variations[tone_name] = new_rgb
            
        return tonal_variations
    
    @staticmethod
    def create_custom_tone(
        base_rgb: List[int],
        saturation_factor: float = 1.0,
        brightness_factor: float = 1.0,
        intensity: float = 1.0
    ) -> List[int]:
        """
        Create a custom tonal variation with specific parameters.
        
        Args:
            base_rgb: Base RGB color [R, G, B] (0-255)
            saturation_factor: Saturation adjustment factor (0.0 to 2.0)
            brightness_factor: Brightness adjustment factor (0.0 to 2.0)
            intensity: Overall intensity of the effect (0.0 to 1.0)
        
        Returns:
            RGB color with applied toning
        """
        # Convert to HSV
        base_hsv = ColorUtils.rgb_to_hsv(base_rgb)
        h, s, v = base_hsv
        
        # Apply intensity-modulated adjustments
        sat_adjustment = 1.0 + (saturation_factor - 1.0) * intensity
        bright_adjustment = 1.0 + (brightness_factor - 1.0) * intensity
        
        # Apply adjustments
        new_s = int(np.clip(s * sat_adjustment, 0, 255))
        new_v = int(np.clip(v * bright_adjustment, 0, 255))
        
        # Convert back to RGB
        new_hsv = [h, new_s, new_v]
        return ColorUtils.hsv_to_rgb(new_hsv)
    
    @staticmethod
    def get_color_info(rgb: List[int]) -> Dict[str, any]:
        """
        Get comprehensive information about a color.
        
        Args:
            rgb: RGB color [R, G, B] (0-255)
            
        Returns:
            Dictionary with color information
        """
        hsv = ColorUtils.rgb_to_hsv(rgb)
        
        # Calculate color properties
        brightness = sum(rgb) / (3 * 255)  # Normalized brightness
        saturation = hsv[1] / 255  # Normalized saturation
        
        # Determine color temperature (rough estimation)
        r, g, b = rgb
        if r > g and r > b:
            temp = "warm"
        elif b > r and b > g:
            temp = "cool"
        else:
            temp = "neutral"
        
        return {
            "rgb": rgb,
            "hsv": hsv,
            "brightness": round(brightness, 2),
            "saturation": round(saturation, 2),
            "temperature": temp,
            "hex": f"#{r:02x}{g:02x}{b:02x}"
        }

    @staticmethod
    def create_color_gradient(
        start_rgb: List[int], 
        end_rgb: List[int], 
        steps: int = 10
    ) -> List[List[int]]:
        """
        Create a gradient between two colors.
        
        Args:
            start_rgb: Start RGB color [R, G, B] (0-255)
            end_rgb: End RGB color [R, G, B] (0-255)
            steps: Number of gradient steps
            
        Returns:
            List of RGB colors in the gradient
        """
        # Convert to numpy arrays
        start = np.array(start_rgb)
        end = np.array(end_rgb)
        
        # Create gradient
        gradient = []
        for i in range(steps):
            # Linear interpolation
            t = i / (steps - 1) if steps > 1 else 0
            color = np.round(start * (1 - t) + end * t).astype(int)
            gradient.append(color.tolist())
            
        return gradient
    
    @staticmethod
    def validate_rgb(rgb: List[int]) -> bool:
        """
        Validate RGB color values.
        
        Args:
            rgb: RGB color [R, G, B]
            
        Returns:
            True if valid, False otherwise
        """
        if not isinstance(rgb, list) or len(rgb) != 3:
            return False
            
        return all(isinstance(c, int) and 0 <= c <= 255 for c in rgb)
    
    @staticmethod
    def dominant_colors(image: np.ndarray, k: int = 5, mask: np.ndarray = None) -> List[Tuple[List[int], float]]:
        """
        Extract dominant colors from an image.
        
        Args:
            image: BGR image
            k: Number of dominant colors to extract
            mask: Optional mask to limit color extraction to specific areas
            
        Returns:
            List of (RGB color, percentage)
        """
        # Reshape image for clustering
        pixels = image.reshape(-1, 3).astype(np.float32)
        
        # Apply mask if provided
        if mask is not None:
            flat_mask = mask.flatten() > 0
            pixels = pixels[flat_mask]
        
        # Must have at least k pixels
        if len(pixels) < k:
            return []
            
        # Apply k-means clustering
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
        _, labels, centers = cv2.kmeans(pixels, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        
        # Count occurrences of each cluster
        counts = np.bincount(labels.flatten())
        
        # Sort by frequency
        indices = np.argsort(counts)[::-1]
        total_pixels = len(labels)
        
        # Create result list
        result = []
        for i in indices:
            percentage = counts[i] / total_pixels
            # Convert BGR to RGB
            center = centers[i].astype(int)[::-1]  # BGR to RGB
            result.append((center.tolist(), percentage))
            
        return result
    
    @staticmethod
    def generate_complementary_colors(rgb: List[int]) -> List[List[int]]:
        """
        Generate complementary and related colors for a given RGB color.
        
        Args:
            rgb: RGB color [R, G, B] (0-255)
            
        Returns:
            List of related colors: [complementary, analogous1, analogous2, triadic1, triadic2]
        """
        # Convert to HSV for easier color manipulation
        hsv = ColorUtils.rgb_to_hsv(rgb)
        h, s, v = hsv
        
        # Calculate related colors in HSV
        complementary_h = (h + 90) % 180  # 180 degrees in OpenCV HSV (H range: 0-179)
        analogous1_h = (h + 30) % 180     # 60 degrees
        analogous2_h = (h - 30) % 180     # -60 degrees
        triadic1_h = (h + 60) % 180       # 120 degrees
        triadic2_h = (h - 60) % 180       # -120 degrees
        
        # Convert back to RGB
        complementary = ColorUtils.hsv_to_rgb([complementary_h, s, v])
        analogous1 = ColorUtils.hsv_to_rgb([analogous1_h, s, v])
        analogous2 = ColorUtils.hsv_to_rgb([analogous2_h, s, v])
        triadic1 = ColorUtils.hsv_to_rgb([triadic1_h, s, v])
        triadic2 = ColorUtils.hsv_to_rgb([triadic2_h, s, v])
        
        return [complementary, analogous1, analogous2, triadic1, triadic2] 