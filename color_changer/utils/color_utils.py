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