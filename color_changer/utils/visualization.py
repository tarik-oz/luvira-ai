"""
Visualization utilities for hair color change previews.
"""

import cv2
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Tuple, Dict, Optional

from color_changer.utils.image_utils import ImageUtils


class Visualizer:
    """
    Utilities for visualizing hair color change results.
    """
    
    @staticmethod
    def create_multi_image_comparison(
        images_and_results: List[Tuple[str, np.ndarray, Dict[str, np.ndarray]]],
        selected_colors: Optional[List[str]] = None
    ) -> None:
        """
        Create comparison grid for multiple images.
        
        Args:
            images_and_results: List of (image_name, original_image, results_dict)
            selected_colors: List of color names to include, or None for all
        """
        n_images = len(images_and_results)
        
        # If no colors specified, use all available in the first result
        if selected_colors is None and n_images > 0:
            selected_colors = list(images_and_results[0][2].keys())
            
        n_colors = len(selected_colors) if selected_colors else 0
        
        if n_images == 0 or n_colors == 0:
            print("No images or colors to display")
            return
            
        # Create figure with columns = 1 (original) + n_colors, and rows = n_images
        fig, axes = plt.subplots(n_images, n_colors + 1, figsize=(3 * (n_colors + 1), 3 * n_images))
        
        # Handle the case with a single image
        if n_images == 1:
            axes = axes.reshape(1, -1)
            
        # Plot each image and its results
        for img_idx, (image_name, original_image, results) in enumerate(images_and_results):
            # Plot original
            original_rgb = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
            axes[img_idx][0].imshow(original_rgb)
            axes[img_idx][0].set_title(f"Original\n{image_name}")
            axes[img_idx][0].axis("off")
            
            # Plot selected color results
            for color_idx, color_name in enumerate(selected_colors):
                if color_name in results:
                    axes[img_idx][color_idx + 1].imshow(results[color_name])
                    axes[img_idx][color_idx + 1].set_title(color_name)
                else:
                    # If this color was not processed for this image
                    axes[img_idx][color_idx + 1].text(0.5, 0.5, "Not Available", 
                                                    ha='center', va='center')
                axes[img_idx][color_idx + 1].axis("off")
                
        plt.tight_layout()
        plt.show()
        
    @staticmethod
    def visualize_preview_results(preview_results_data: List[Tuple[str, List[Tuple[str, str]]]]) -> None:
        """
        Visualize the results of batch preview.
        
        Args:
            preview_results_data: Output from PreviewRunner.run_batch_preview()
        """
        if not preview_results_data:
            print("No preview results to visualize")
            return
        
        # Extract all color names used in previews
        all_colors = set()
        for _, color_results in preview_results_data:
            for color_name, _ in color_results:
                all_colors.add(color_name)
                
        # Sort color names alphabetically
        color_names = sorted(all_colors)
        
        # Prepare data for multi-image comparison
        comparison_data = []
        for image_file, color_results in preview_results_data:
            # Load original image
            image_path = f"test_images/{image_file}"
            original_image = ImageUtils.load_image(image_path)
            
            if original_image is None:
                print(f"Failed to load {image_path}, skipping")
                continue
                
            # Prepare results dict
            results_dict = {}
            for color_name, result_path in color_results:
                result_image = ImageUtils.load_image(result_path)
                if result_image is not None:
                    # Convert BGR to RGB for display
                    results_dict[color_name] = cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)
                    
            comparison_data.append((image_file, original_image, results_dict))
            
        # Create visualization
        Visualizer.create_multi_image_comparison(comparison_data, color_names) 