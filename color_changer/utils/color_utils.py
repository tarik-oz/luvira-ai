"""
Color utilities for hair color change operations.
"""

import cv2
import numpy as np
from typing import List

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
        intensity: float = 1.0,
        hue_offset_degrees: float = 0.0,
    ) -> List[int]:
        """
        Create a custom tonal variation with specific parameters.
        
        Args:
            base_rgb: Base RGB color [R, G, B] (0-255)
            saturation_factor: Saturation adjustment factor (0.0 to 2.0)
            brightness_factor: Brightness adjustment factor (0.0 to 2.0)
            intensity: Overall intensity of the effect (0.0 to 1.0)
            hue_offset_degrees: Optional hue offset in degrees (-180..+180). Positive values shift towards cooler hues.
        
        Returns:
            RGB color with applied toning
        """
        # Convert to HSV
        base_hsv = ColorUtils.rgb_to_hsv(base_rgb)
        h, s, v = base_hsv
        
        # Apply intensity-modulated adjustments for S/V
        sat_adjustment = 1.0 + (saturation_factor - 1.0) * intensity
        bright_adjustment = 1.0 + (brightness_factor - 1.0) * intensity
        
        # Apply adjustments
        new_s = int(np.clip(s * sat_adjustment, 0, 255))
        new_v = int(np.clip(v * bright_adjustment, 0, 255))
        
        # Apply hue offset if given (degrees -> OpenCV hue units)
        if abs(hue_offset_degrees) > 1e-6:
            # OpenCV H is 0..179 which maps to 0..360 degrees (x2)
            hue_offset_cv = hue_offset_degrees / 2.0
            new_h = (float(h) + float(hue_offset_cv)) % 180.0
        else:
            new_h = float(h)
        
        # Convert back to RGB
        new_hsv = [int(new_h), new_s, new_v]
        return ColorUtils.hsv_to_rgb(new_hsv)
    
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
        color_name_clean = color_name.strip()
        _, found_name = ColorUtils.find_color_by_name(color_name_clean)
        if not found_name:
            print(f"Error: Color '{color_name}' not found.\nTip: Check for typos or extra spaces.")
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
        color_name_clean = color_name.strip().lower()
        for rgb, name in COLORS:
            if name.lower() == color_name_clean:
                return rgb, name
        return None, None
    
    @staticmethod
    def get_available_colors():
        """Get list of all available colors."""
        return COLORS.copy()
