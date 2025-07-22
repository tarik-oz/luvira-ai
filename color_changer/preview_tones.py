#!/usr/bin/env python
"""
Hair color tone preview script.

This script provides a command-line interface for previewing tone variations 
of hair colors. It allows selecting specific images, one base color, and 
multiple tones to apply, and visualizes the results.
"""

import os
import sys
import argparse

# Import directly from local files
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import core modules directly
from color_changer.core.color_transformer import ColorTransformer
from color_changer.config.color_config import (
    COLORS, CUSTOM_TONES, PREVIEW_IMAGES_DIR, PREVIEW_RESULTS_DIR, DEFAULT_MODEL_PATH
)

# Create local imports for runner and visualizer
from color_changer.utils.visualization import Visualizer
from color_changer.utils.image_utils import ImageUtils
from color_changer.utils.preview_utils import (
    find_image_files, validate_image_list, get_valid_images_with_masks
)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Preview hair color tone variations on images')
    
    parser.add_argument('--images', nargs='+', default=[],
                        help='Specific image files to process (default: all images in directory)')
    
    parser.add_argument('--images-dir', type=str, default=str(PREVIEW_IMAGES_DIR),
                        help=f'Directory containing images (uses PREVIEW_IMAGES_DIR from config if not provided)')
    
    parser.add_argument('--results-dir', type=str, default=str(PREVIEW_RESULTS_DIR),
                        help=f'Directory to save results (uses PREVIEW_RESULTS_DIR from config if not provided)')
    
    parser.add_argument('--color', type=str,
                        help='Base color to apply tones to (required unless using list options)')
    
    parser.add_argument('--tones', nargs='+',
                        help='Space-separated list of tones to apply (default: all tones for the color)')
    
    parser.add_argument('--no-visualization', action='store_true',
                        help='Disable visualization of results')
    
    parser.add_argument('--list-colors', action='store_true',
                        help='List available color names and exit')
                        
    parser.add_argument('--list-tones', type=str,
                        help='List available tones for a specific color and exit')
                        
    parser.add_argument('--model', type=str, default=str(DEFAULT_MODEL_PATH),
                        help=f'Path to trained model (uses DEFAULT_MODEL_PATH from config if not provided)')
                        
    parser.add_argument('--use-existing-masks', action='store_true',
                        help='Use existing masks instead of generating new ones')
                        
    parser.add_argument('--device', type=str, choices=['auto', 'cpu', 'cuda'], default='auto',
                        help='Device to use for prediction (default: auto)')
    
    return parser.parse_args()


def find_color_info(color_name):
    """Find color info from COLORS config."""
    for rgb, name in COLORS:
        if name.lower() == color_name.lower():
            return rgb, name
    return None, None


def main():
    """Main entry point for the preview script."""
    args = parse_arguments()
    
    # If --list-colors is specified, print available colors and exit
    if args.list_colors:
        print("Available colors:")
        for rgb, name in COLORS:
            tone_count = len(CUSTOM_TONES.get(name, {}))
            print(f"  {name}: RGB{rgb} ({tone_count} tones available)")
        sys.exit(0)
    
    # If --list-tones is specified, print available tones for color and exit
    if args.list_tones:
        color_rgb, color_name = find_color_info(args.list_tones)
        if not color_name:
            print(f"Error: Color '{args.list_tones}' not found.")
            print("Available colors:", [name for _, name in COLORS])
            sys.exit(1)
        
        if color_name not in CUSTOM_TONES:
            print(f"No tones available for {color_name}")
        else:
            print(f"Available tones for {color_name}:")
            for tone_name, config in CUSTOM_TONES[color_name].items():
                print(f"  {tone_name:12}: {config['description']}")
        sys.exit(0)
    
    # Check if color is required
    if not args.color:
        print("Error: --color is required (except when using --list-colors or --list-tones)")
        print("Use --list-colors to see all available colors.")
        sys.exit(1)
    
    # Validate base color
    base_color_rgb, base_color_name = find_color_info(args.color)
    if not base_color_name:
        print(f"Error: Color '{args.color}' not found.")
        print("Available colors:", [name for _, name in COLORS])
        print("Use --list-colors to see all available colors.")
        sys.exit(1)
    
    # Get available tones for the color
    if base_color_name not in CUSTOM_TONES:
        print(f"Error: No tones available for {base_color_name}")
        sys.exit(1)
    
    available_tones = list(CUSTOM_TONES[base_color_name].keys())
    
    # Filter tones if specified
    if args.tones:
        selected_tones = []
        for tone_name in args.tones:
            if tone_name in available_tones:
                selected_tones.append(tone_name)
            else:
                print(f"Warning: Tone '{tone_name}' not available for {base_color_name}, ignoring.")
        
        if not selected_tones:
            print(f"No valid tones specified, using all available tones for {base_color_name}.")
            selected_tones = available_tones
    else:
        selected_tones = available_tones
    
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
    print(f"\nApplying {len(selected_tones)} tones of {base_color_name} on {len(valid_images)} images:")
    print(f"  Base Color: {base_color_name} RGB{base_color_rgb}")
    print(f"  Tones: {', '.join(selected_tones)}")
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
        
        # Apply each tone
        image_results = []
        base_name = os.path.splitext(img_name)[0]
        
        # Always include base color for comparison
        try:
            base_result = transformer.change_hair_color(image, mask, base_color_rgb)
            out_path = os.path.join(args.results_dir, f"{base_name}_{base_color_name.lower()}_base.png")
            ImageUtils.save_image(base_result, out_path, convert_to_bgr=True)
            image_results.append((f"{base_color_name} (base)", out_path))
            print(f"Successfully applied base {base_color_name} to {img_name}")
        except Exception as e:
            print(f"Failed to apply base {base_color_name} to {img_name}: {str(e)}")
        
        # Apply tones
        for tone_name in selected_tones:
            try:
                # Apply tone transformation
                result = transformer.apply_color_with_tone(
                    image, mask, base_color_rgb, base_color_name, tone_name
                )
                
                # Save result
                out_path = os.path.join(args.results_dir, f"{base_name}_{base_color_name.lower()}_{tone_name}.png")
                ImageUtils.save_image(result, out_path, convert_to_bgr=True)
                
                image_results.append((f"{base_color_name} ({tone_name})", out_path))
                print(f"Successfully applied {base_color_name} {tone_name} tone to {img_name}")
                
            except Exception as e:
                print(f"Failed to apply {base_color_name} {tone_name} tone to {img_name}: {str(e)}")
        
        if image_results:
            results.append((img_name, image_results))
    
    # Visualize results if requested
    if not args.no_visualization and results:
        print("\nVisualizing results...")
        Visualizer.visualize_preview_results(results)
    
    print("\nDone!")


if __name__ == "__main__":
    main()
