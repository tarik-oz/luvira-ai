"""
Preview runner for hair color change operations.
"""

import os
import cv2
from typing import List, Tuple, Optional, TYPE_CHECKING

from color_changer.config.color_config import COLORS

# Lazy import to avoid circular dependency
if TYPE_CHECKING:
    from color_changer.core.color_transformer import ColorTransformer


class PreviewRunner:
    """
    Utility for running color change previews on images.
    """
    
    def __init__(self, images_dir: str = "test_images", results_dir: str = "test_results"):
        """
        Initialize the preview runner.
        
        Args:
            images_dir: Directory containing preview images
            results_dir: Directory to save preview results
        """
        self.images_dir = images_dir
        self.results_dir = results_dir
        self._transformer = None  # Lazy initialization
        
        # Create results directory if it doesn't exist
        os.makedirs(results_dir, exist_ok=True)
    
    @property
    def transformer(self):
        """Lazy initialization of ColorTransformer to avoid circular imports."""
        if self._transformer is None:
            from color_changer.core.color_transformer import ColorTransformer
            self._transformer = ColorTransformer()
        return self._transformer
    
    def find_preview_images(self) -> List[str]:
        """
        Find all preview images in the images directory.
        
        Returns:
            List of image filenames
        """
        image_files = [f for f in os.listdir(self.images_dir) 
                      if f.endswith(".jpeg") or f.endswith(".jpg") or f.endswith(".png")]
        return sorted(image_files)
    
    def find_mask_for_image(self, image_filename: str) -> Optional[str]:
        """
        Find the mask file for a given image.
        
        Args:
            image_filename: Image filename
            
        Returns:
            Mask filename or None if not found
        """
        base = os.path.splitext(image_filename)[0]
        mask_path = os.path.join(self.images_dir, f"{base}_prob_mask.png")
        if not os.path.exists(mask_path):
            mask_path = os.path.join(self.images_dir, f"{base}_mask.png")
            if not os.path.exists(mask_path):
                return None
        return mask_path
    
    def process_image(self, image_filename: str, colors_to_apply: List[Tuple[List[int], str]]) -> List[Tuple[str, str]]:
        """
        Apply color changes to a single image with multiple colors.
        
        Args:
            image_filename: Image filename
            colors_to_apply: List of colors to apply, each as (RGB, name)
            
        Returns:
            List of (color_name, output_path) for successful transformations
        """
        results = []
        
        # Load image and mask
        image_path = os.path.join(self.images_dir, image_filename)
        mask_path = self.find_mask_for_image(image_filename)
        
        if mask_path is None:
            print(f"No mask found for {image_filename}, skipping.")
            return results
            
        image = cv2.imread(image_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        if image is None or mask is None:
            print(f"Failed to load {image_path} or its mask, skipping.")
            return results
            
        # Apply each color
        for rgb_color, color_name in colors_to_apply:
            try:
                # Apply color transformation
                result = self.transformer.change_hair_color(image, mask, rgb_color)
                
                # Save result
                base_name = os.path.splitext(image_filename)[0]
                out_path = os.path.join(self.results_dir, f"{base_name}_to_{color_name.lower()}.png")
                cv2.imwrite(out_path, cv2.cvtColor(result, cv2.COLOR_RGB2BGR))
                
                results.append((color_name, out_path))
                print(f"Successfully applied {color_name} to {image_filename}")
                
            except Exception as e:
                print(f"Failed to apply {color_name} to {image_filename}: {str(e)}")
                
        return results
    
    def run_batch_preview(self, selected_colors: List[Tuple[List[int], str]] = None) -> List[Tuple[str, List[Tuple[str, str]]]]:
        """
        Run batch preview on all images with selected colors.
        
        Args:
            selected_colors: List of colors to apply, or None to use all predefined colors
            
        Returns:
            List of (image_name, [(color_name, output_path), ...]) for all successful transformations
        """
        # Use all colors if none specified
        colors_to_apply = selected_colors if selected_colors is not None else COLORS
        
        # Find all preview images
        image_files = self.find_preview_images()
        
        # Process each image
        results = []
        for image_file in image_files:
            image_results = self.process_image(image_file, colors_to_apply)
            if image_results:
                results.append((image_file, image_results))
                
        return results 