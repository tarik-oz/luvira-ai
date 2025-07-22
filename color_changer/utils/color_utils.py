"""
Color utilities for hair color change operations.
"""

import cv2
import numpy as np
from typing import List, Dict

from color_changer.config.color_config import CUSTOM_TONES

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
        color_name: str = None
    ) -> Dict[str, List[int]]:
        """
        Generate tonal variations of a base color using color-specific tones.
        
        Args:
            base_rgb: Base RGB color [R, G, B] (0-255)
            color_name: Name of the color to get specific tones for
        
        Returns:
            Dictionary mapping tone names to RGB colors
        """
        if color_name is None or color_name not in CUSTOM_TONES:
            # Return empty dict if no color specified or invalid color
            return {}
            
        tone_configs = CUSTOM_TONES[color_name]
        
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
    def get_available_tones() -> Dict[str, Dict]:
        """
        Get all available tone types and their configurations.
        
        Returns:
            Dictionary of tone configurations
        """
        return CUSTOM_TONES.copy()
    
    @staticmethod
    def get_tone_info(base_color: List[int], color_name: str, tone_name: str) -> Dict:
        """
        Get information about a specific tone of a color.
        
        Args:
            base_color: Base RGB color [R, G, B] (0-255)
            color_name: Name of the base color
            tone_name: Name of the tone
            
        Returns:
            Dictionary with tone information
        """
        
        if color_name not in CUSTOM_TONES:
            raise ValueError(f"Invalid color name: {color_name}. Available: {list(CUSTOM_TONES.keys())}")
            
        if tone_name not in CUSTOM_TONES[color_name]:
            raise ValueError(f"Invalid tone name: {tone_name}. Available for {color_name}: {list(CUSTOM_TONES[color_name].keys())}")
        
        tone_config = CUSTOM_TONES[color_name][tone_name]
        
        # Generate the toned color
        toned_color = ColorUtils.create_custom_tone(
            base_color,
            saturation_factor=tone_config["saturation_factor"],
            brightness_factor=tone_config["brightness_factor"],
            intensity=1.0
        )
        
        # Get color information
        base_info = ColorUtils.get_color_info(base_color)
        toned_info = ColorUtils.get_color_info(toned_color)
        
        return {
            "color_name": color_name,
            "tone_name": tone_name,
            "description": tone_config["description"],
            "base_color": {
                "rgb": base_color,
                "hex": base_info["hex"],
                "info": base_info
            },
            "toned_color": {
                "rgb": toned_color,
                "hex": toned_info["hex"],
                "info": toned_info
            },
            "adjustments": {
                "saturation_factor": tone_config["saturation_factor"],
                "brightness_factor": tone_config["brightness_factor"]
            }
        }