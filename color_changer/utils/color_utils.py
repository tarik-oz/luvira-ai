"""
Color utilities for hair color change operations.
"""

import cv2
import numpy as np
from typing import List, Dict

from color_changer.config.color_config import CUSTOM_TONES, COLORS

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
    def get_available_tones() -> Dict[str, Dict]:
        """
        Get all available tone types and their configurations.
        
        Returns:
            Dictionary of tone configurations
        """
        return CUSTOM_TONES.copy()
    
    @staticmethod
    def list_colors():
        """Print available colors with tone counts."""
        print("Available colors:")
        for rgb, name in COLORS:
            tone_count = len(CUSTOM_TONES.get(name, {}))
            print(f"  {name}: RGB{rgb} ({tone_count} tones available)")
    
    @staticmethod
    def list_tones_for_color(color_name: str) -> bool:
        """
        Print available tones for a specific color.
        
        Args:
            color_name: Name of the color
            
        Returns:
            bool: True if color found and tones listed, False otherwise
        """
        # Find color
        color_rgb, found_name = ColorUtils.find_color_by_name(color_name)
        if not found_name:
            print(f"Error: Color '{color_name}' not found.")
            print("Available colors:", [name for _, name in COLORS])
            return False
        
        if found_name not in CUSTOM_TONES:
            print(f"No tones available for {found_name}")
        else:
            print(f"Available tones for {found_name}:")
            for tone_name, config in CUSTOM_TONES[found_name].items():
                print(f"  {tone_name:12}: {config['description']}")
        return True
    
    @staticmethod
    def find_color_by_name(color_name: str):
        """
        Find color RGB and name by color name.
        
        Args:
            color_name: Name of the color to find
            
        Returns:
            Tuple: (rgb, name) or (None, None) if not found
        """
        for rgb, name in COLORS:
            if name.lower() == color_name.lower():
                return rgb, name
        return None, None
    
    @staticmethod
    def get_available_colors():
        """Get list of all available colors."""
        return COLORS.copy()
