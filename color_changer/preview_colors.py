#!/usr/bin/env python
"""
Hair color change preview script.

This script provides a command-line interface for previewing the hair color change functionality.
It allows selecting specific images and colors to apply, and visualizes the results.
"""

import os
import sys
import argparse

# Import directly from local files
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import core modules directly
from color_changer.core.color_transformer import ColorTransformer
from color_changer.config.color_config import (
    COLORS, PREVIEW_IMAGES_DIR, PREVIEW_RESULTS_DIR, DEFAULT_MODEL_PATH
)

# Create local imports for runner and visualizer
from color_changer.utils.visualization import Visualizer
from color_changer.utils.image_utils import ImageUtils
from color_changer.utils.preview_utils import (
    generate_hair_masks, find_existing_masks, find_image_files, 
    validate_image_list, get_valid_images_with_masks
)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Preview hair color changes on images')
    
    parser.add_argument('--images', nargs='+', default=[],
                        help='Specific image files to process (default: all images in directory)')
    
    parser.add_argument('--images-dir', type=str, default=str(PREVIEW_IMAGES_DIR),
                        help=f'Directory containing images (uses PREVIEW_IMAGES_DIR from config if not provided)')
    
    parser.add_argument('--results-dir', type=str, default=str(PREVIEW_RESULTS_DIR),
                        help=f'Directory to save results (uses PREVIEW_RESULTS_DIR from config if not provided)')
    
    parser.add_argument('--colors', nargs='+',
                        help='Space-separated list of colors to apply (default: all colors)')
    
    parser.add_argument('--no-visualization', action='store_true',
                        help='Disable visualization of results')
    
    parser.add_argument('--list-colors', action='store_true',
                        help='List available color names and exit')
                        
    parser.add_argument('--model', type=str, default=str(DEFAULT_MODEL_PATH),
                        help=f'Path to trained model (uses DEFAULT_MODEL_PATH from config if not provided)')
                        
    parser.add_argument('--use-existing-masks', action='store_true',
                        help='Use existing masks instead of generating new ones')
                        
    parser.add_argument('--device', type=str, choices=['auto', 'cpu', 'cuda'], default='auto',
                        help='Device to use for prediction (default: auto)')
    
    return parser.parse_args()

def main():
    """Main entry point for the preview script."""
    args = parse_arguments()
    
    # Get all colors
    all_colors = COLORS
    
    # If --list-colors is specified, print available colors and exit
    if args.list_colors:
        print("Available colors:")
        for color, name in all_colors:
            print(f"  {name}: {color}")
        sys.exit(0)
    
    # Filter colors if specified
    if args.colors:
        selected_colors = []
        for color_name in args.colors:
            color_name_lower = color_name.lower()
            found = False
            for rgb, name in all_colors:
                if name.lower() == color_name_lower:
                    selected_colors.append((rgb, name))
                    found = True
                    break
            if not found:
                print(f"Warning: Color '{color_name}' not found, ignoring.")
        
        if not selected_colors:
            print("No valid colors specified, using all colors.")
            selected_colors = all_colors
    else:
        selected_colors = all_colors
    
    # Find image files
    if args.images:
        # Use specified images
        selected_images = validate_image_list(args.images, args.images_dir)
    else:
        # Find all image files in the directory
        selected_images = find_image_files(args.images_dir)
    
    if not selected_images:
        print(f"No images found in {args.images_dir}")
        sys.exit(1)
    
    # Get masks for the selected images
    valid_images, image_to_mask = get_valid_images_with_masks(
        selected_images, args.images_dir, args.use_existing_masks, 
        args.model, args.device
    )
    if not valid_images:
        print("No valid images with masks found.")
        sys.exit(1)
    
    # Print preview setup
    print(f"\nApplying {len(selected_colors)} colors on {len(valid_images)} images:")
    print(f"  Colors: {', '.join(name for _, name in selected_colors)}")
    print(f"  Images: {', '.join(os.path.basename(img) for img in valid_images)}")
    print(f"  Results will be saved to: {args.results_dir}")
    
    # Create results directory if it doesn't exist
    os.makedirs(args.results_dir, exist_ok=True)
    
    # Create color transformer
    transformer = ColorTransformer()
    
    # Process each image
    results = []
    for img_path in valid_images:
        img_name = os.path.basename(img_path)
        mask_path = image_to_mask[img_path]
        
        # Load image and mask
        image = ImageUtils.load_image(img_path)
        mask = ImageUtils.load_image(mask_path, grayscale=True)
        
        if image is None or mask is None:
            print(f"Failed to load {img_name} or its mask, skipping.")
            continue
        
        # Apply each color
        image_results = []
        for rgb_color, color_name in selected_colors:
            try:
                # Apply color transformation
                result = transformer.change_hair_color(image, mask, rgb_color)
                
                # Save result
                base_name = os.path.splitext(img_name)[0]
                out_path = os.path.join(args.results_dir, f"{base_name}_to_{color_name.lower()}.png")
                ImageUtils.save_image(result, out_path, convert_to_bgr=True)
                
                image_results.append((color_name, out_path))
                print(f"Successfully applied {color_name} to {img_name}")
                
            except Exception as e:
                print(f"Failed to apply {color_name} to {img_name}: {str(e)}")
        
        if image_results:
            results.append((img_name, image_results))
    
    # Visualize results if requested
    if not args.no_visualization and results:
        print("\nVisualizing results...")
        Visualizer.visualize_preview_results(results)
    
    print("\nDone!")


if __name__ == "__main__":
    main() 