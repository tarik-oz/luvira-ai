#!/usr/bin/env python
"""
Test script to apply all colors and their tones to a specific image.
Quick and dirty for testing purposes.
"""

import os
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Add project to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from color_changer.core.color_transformer import ColorTransformer
from color_changer.config.color_config import COLORS, CUSTOM_TONES
from color_changer.utils.image_utils import ImageUtils

def test_all_colors_and_tones(image_path, mask_path=None):
    """Test all colors and tones on a specific image."""
    
    # Load image
    if not os.path.exists(image_path):
        print(f"Error: Image not found: {image_path}")
        return
        
    image = ImageUtils.load_image(image_path)
    if image is None:
        print(f"Error: Could not load image: {image_path}")
        return
    
    # Load or find mask
    if mask_path and os.path.exists(mask_path):
        mask = ImageUtils.load_image(mask_path, grayscale=True)
        print(f"Using provided mask: {mask_path}")
    else:
        # Try to find mask automatically
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        image_dir = os.path.dirname(image_path)
        
        # Try different mask patterns
        mask_patterns = [
            os.path.join(image_dir, f"{base_name}_prob_mask.png"),
            os.path.join(image_dir, f"{base_name}_mask.png"),
            os.path.join('..', 'model', 'test_results', f"{base_name}_prob_mask.png"),
        ]
        
        mask = None
        for pattern in mask_patterns:
            if os.path.exists(pattern):
                mask = ImageUtils.load_image(pattern, grayscale=True)
                print(f"Found mask: {pattern}")
                break
                
        if mask is None:
            print("Warning: No mask found! Creating dummy mask (center region)")
            h, w = image.shape[:2]
            mask = np.zeros((h, w), dtype=np.uint8)
            mask[h//4:3*h//4, w//4:3*w//4] = 255
    
    # Create transformer
    transformer = ColorTransformer()
    
    # Process each color
    all_results = {}  # ordered dict to maintain order
    
    for color_rgb, color_name in COLORS:
        print(f"\nProcessing {color_name}...")
        
        try:
            # Get all tones for this color using optimized method
            results = transformer.change_hair_color_with_all_tones(image, mask, color_name)
            
            # Store base result FIRST
            all_results[f"{color_name}_base"] = results['base_result']
            
            # Then store tone results in a consistent order
            for tone_name in sorted(results['tones'].keys()):  # Sort for consistent order
                tone_result = results['tones'][tone_name]
                if tone_result is not None:
                    all_results[f"{color_name}_{tone_name}"] = tone_result
                    
        except Exception as e:
            print(f"Error processing {color_name}: {e}")
            continue
    
    # Create visualization
    print(f"\nCreating visualization with {len(all_results)} results...")
    
    # Calculate grid size
    total_images = len(all_results) + 1  # +1 for original
    cols = min(6, total_images)  # Max 6 columns
    rows = (total_images + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    if rows == 1:
        axes = axes.reshape(1, -1)
    
    # Flatten axes for easier indexing
    axes_flat = axes.flatten()
    
    # Show original
    original_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    axes_flat[0].imshow(original_rgb)
    axes_flat[0].set_title("Original", fontsize=10)
    axes_flat[0].axis('off')
    
    # Show results
    for idx, (name, result) in enumerate(all_results.items(), 1):
        if idx < len(axes_flat):
            axes_flat[idx].imshow(result)
            axes_flat[idx].set_title(name, fontsize=8)
            axes_flat[idx].axis('off')
    
    # Hide unused subplots
    for idx in range(len(all_results) + 1, len(axes_flat)):
        axes_flat[idx].axis('off')
    
    plt.tight_layout()
    plt.suptitle(f"All Colors & Tones - {os.path.basename(image_path)}", fontsize=14)
    plt.show()
    
    # Save results
    output_dir = "test_all_results"
    os.makedirs(output_dir, exist_ok=True)
    
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    
    for name, result in all_results.items():
        output_path = os.path.join(output_dir, f"{base_name}_{name}.png")
        ImageUtils.save_image(result, output_path, convert_to_bgr=True)
    
    print(f"\nResults saved to: {output_dir}/")
    print(f"Total results: {len(all_results)}")

def main():
    # Test configuration
    test_image = "test_images_2/woman_3.png"  # Change this to your test image
    test_mask = None  # Will auto-find mask
    
    print("=== Color & Tone Test Script ===")
    print(f"Testing image: {test_image}")
    print(f"Available colors: {len(COLORS)}")
    print(f"Total tones: {sum(len(tones) for tones in CUSTOM_TONES.values())}")
    
    test_all_colors_and_tones(test_image, test_mask)

if __name__ == "__main__":
    main()
