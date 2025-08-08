"""
Main color transformer class that orchestrates the hair color transformation process.
"""

import cv2
import numpy as np
from typing import List, Tuple, Dict

from color_changer.transformers.hsv_transformer import HsvTransformer
from color_changer.transformers.blender import Blender
from color_changer.utils.color_utils import ColorUtils
from color_changer.config.color_config import CUSTOM_TONES, COLORS


class ColorTransformer:
    """
    Main class for hair color transformation, orchestrates the process by delegating
    to specialized components for different transformation steps.
    """
    
    def __init__(self):
        self.hsv_transformer = HsvTransformer()
        self.blender = Blender()
    
    def change_hair_color(self, image: np.ndarray, mask: np.ndarray, color_input) -> np.ndarray:
        """
        Natural hair recoloring that preserves original texture and lighting.
        Focuses on subtle, realistic color changes.
        
        Args:
            image: Original image (BGR format, np.ndarray)
            mask: Grayscale mask (0-255, np.ndarray) 
            color_input: Color name (str) or RGB values (List[int])
        
        Returns:
            Recolored image (RGB format, np.ndarray)
        """
        # Convert color input to RGB
        if isinstance(color_input, str):
            target_rgb = self._get_rgb_from_color_name(color_input)
        else:
            target_rgb = color_input  # Already RGB list
        
        # Preprocess inputs
        image_rgb, image_float, mask_normalized, mask_3ch, target_rgb_norm, target_hsv = \
            self._preprocess_inputs(image, mask, target_rgb)
        
        # Check if hair is detected
        if np.sum(mask_normalized > 0.1) == 0:
            return image_rgb  # No hair detected, return original
        
        # Analyze hair characteristics
        alpha, saturation_factor, brightness_adjustment = \
            self._analyze_hair_characteristics(image_float, mask_normalized, target_rgb_norm)
        
        # Convert to HSV for better color control
        image_hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV).astype(np.float32)
        
        # Apply HSV transformations
        result_hsv = self.hsv_transformer.apply_hsv_transformations(
            image_hsv, mask_normalized, target_hsv, alpha, saturation_factor, brightness_adjustment
        )
        
        # Convert back to RGB
        result_rgb = cv2.cvtColor(result_hsv.astype(np.uint8), cv2.COLOR_HSV2RGB).astype(np.float32) / 255.0
        
        # Apply natural blending
        result = self.blender.apply_natural_blending(image_float, result_rgb, mask_3ch, alpha)
        
        # Convert back to RGB and return
        result_rgb = np.clip(result * 255, 0, 255).astype(np.uint8)
        return result_rgb  # Return RGB format for web API
    
    def apply_color_with_tone(
        self, 
        image: np.ndarray, 
        mask: np.ndarray, 
        color_name: str,
        tone_name: str
    ) -> np.ndarray:
        """
        Apply hair color change with specific toning.
        
        Args:
            image: Original image (BGR format, np.ndarray)
            mask: Grayscale mask (0-255, np.ndarray)
            color_name: Name of the color (e.g., "Blonde", "Brown", etc.)
            tone_name: Name of the tone (e.g., "golden", "ash", etc.)
            
        Returns:
            Recolored image (RGB format, np.ndarray)
        """
        # Get base color RGB from config
        base_color_rgb = self._get_rgb_from_color_name(color_name)
        
        # Get tone configuration
        if color_name not in CUSTOM_TONES:
            raise ValueError(f"Invalid color name: {color_name}. Available: {list(CUSTOM_TONES.keys())}")
            
        if tone_name not in CUSTOM_TONES[color_name]:
            raise ValueError(f"Invalid tone name: {tone_name}. Available for {color_name}: {list(CUSTOM_TONES[color_name].keys())}")
        
        tone_config = CUSTOM_TONES[color_name][tone_name]
        
        # Generate toned color
        toned_color = ColorUtils.create_custom_tone(
            base_color_rgb,
            saturation_factor=tone_config["saturation_factor"],
            brightness_factor=tone_config["brightness_factor"],
            intensity=1.0
        )
        
        # Apply the toned color (now passing RGB values instead of color name)
        return self.change_hair_color(image, mask, toned_color)

    def change_hair_color_with_all_tones(
        self, 
        image: np.ndarray, 
        mask: np.ndarray, 
        color_name: str
    ) -> Dict[str, np.ndarray]:
        """
        Apply hair color change with base color and all available tones (MAIN ENDPOINT).
        
        This is the primary function for the main API endpoint. It efficiently processes
        the image once and generates both the base color result and all tone variations.
        
        Args:
            image: Original image (BGR format, np.ndarray)
            mask: Grayscale mask (0-255, np.ndarray)
            color_name: Name of the color from COLORS (e.g., "Blonde", "Brown", etc.)
            
        Returns:
            Dictionary with base result and all tones:
            {
                'base_result': np.ndarray,  # Base color transformation
                'tones': {
                    'golden': np.ndarray,   # Each tone transformation
                    'ash': np.ndarray,
                    ...
                }
            }
        """
        # Get base color RGB from config
        base_color_rgb = self._get_rgb_from_color_name(color_name)
        
        # Check if color has tone configurations
        if color_name not in CUSTOM_TONES:
            # If no tones defined, just return base result
            base_result = self.change_hair_color(image, mask, color_name)
            return {
                'base_result': base_result,
                'tones': {}
            }
        
        # Preprocess inputs once (performance optimization)
        image_rgb, image_float, mask_normalized, mask_3ch, target_rgb, target_hsv = \
            self._preprocess_inputs(image, mask, base_color_rgb)
        
        # Check if hair is detected
        if np.sum(mask_normalized > 0.1) == 0:
            return {
                'base_result': image_rgb,  # No hair detected
                'tones': {}
            }
        
        # Analyze hair characteristics once
        alpha, saturation_factor, brightness_adjustment = \
            self._analyze_hair_characteristics(image_float, mask_normalized, target_rgb)
        
        # Convert to HSV once
        image_hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV).astype(np.float32)
        
        results = {}
        
        # Generate base result
        result_hsv = self.hsv_transformer.apply_hsv_transformations(
            image_hsv, mask_normalized, target_hsv, alpha, saturation_factor, brightness_adjustment
        )
        result_rgb = cv2.cvtColor(result_hsv.astype(np.uint8), cv2.COLOR_HSV2RGB).astype(np.float32) / 255.0
        base_result = self.blender.apply_natural_blending(image_float, result_rgb, mask_3ch, alpha)
        results['base_result'] = np.clip(base_result * 255, 0, 255).astype(np.uint8)
        
        # Generate all tones efficiently
        results['tones'] = {}
        for tone_name, tone_config in CUSTOM_TONES[color_name].items():
            try:
                # Generate toned color
                toned_color = ColorUtils.create_custom_tone(
                    base_color_rgb,
                    saturation_factor=tone_config["saturation_factor"],
                    brightness_factor=tone_config["brightness_factor"],
                    intensity=1.0
                )
                
                # Convert toned color to HSV
                toned_hsv = cv2.cvtColor(np.uint8([[toned_color]]), cv2.COLOR_RGB2HSV)[0][0].astype(np.float32)
                
                # Apply HSV transformations for this tone
                tone_result_hsv = self.hsv_transformer.apply_hsv_transformations(
                    image_hsv, mask_normalized, toned_hsv, alpha, saturation_factor, brightness_adjustment
                )
                tone_result_rgb = cv2.cvtColor(tone_result_hsv.astype(np.uint8), cv2.COLOR_HSV2RGB).astype(np.float32) / 255.0
                tone_result = self.blender.apply_natural_blending(image_float, tone_result_rgb, mask_3ch, alpha)
                results['tones'][tone_name] = np.clip(tone_result * 255, 0, 255).astype(np.uint8)
                
            except Exception as e:
                print(f"Error generating tone '{tone_name}': {e}")
                results['tones'][tone_name] = None
        
        return results
    
    def _preprocess_inputs(self, image: np.ndarray, mask: np.ndarray, target_color: List[int]) -> Tuple:
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
    
    def _analyze_hair_characteristics(self, image_float: np.ndarray, mask_normalized: np.ndarray, target_rgb: np.ndarray) -> Tuple[float, float, float]:
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

    def _get_rgb_from_color_name(self, color_name: str) -> List[int]:
        """
        Convert a color name to its corresponding RGB values from the COLORS config.

        Args:
            color_name (str): Name of the color (e.g., "Blonde", "Brown", etc.)

        Returns:
            List[int]: RGB values [R, G, B]

        Raises:
            ValueError: If color name is not found in COLORS.

        Notes:
            - Strips leading/trailing whitespace from color_name.
            - Case-insensitive comparison.
        """
        search_name = color_name.strip().lower()
        # COLORS: List of (rgb, name)
        for color_rgb, name in COLORS:
            if name.lower() == search_name:
                return color_rgb
        # If not found, show available colors
        available_colors = [name for _, name in COLORS]
        raise ValueError(
            f"Color '{color_name}' not found.\n"
            f"Tip: Check for typos or extra spaces.\n"
            f"Available colors: {available_colors}"
        )