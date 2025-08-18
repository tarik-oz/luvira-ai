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
from color_changer.config.color_config import (
    PREVIEW_IMAGES_DIR, PREVIEW_RESULTS_DIR, DEFAULT_MODEL_PATH
)

# Create local imports for runner and visualizer
from color_changer.utils.visualization import Visualizer
from color_changer.utils.image_utils import ImageUtils
from color_changer.utils.preview_utils import (
    find_image_files, validate_image_list, get_valid_images_with_masks,
    handle_list_commands, validate_single_color, select_tones_for_color, process_images_with_tones
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


def main():
    """Main entry point for the preview script."""
    args = parse_arguments()
    
    # Handle list commands
    if handle_list_commands(args):
        sys.exit(0)
    
    # Check if color is required
    if not args.color:
        print("Error: --color is required (except when using --list-colors or --list-tones)")
        print("Use --list-colors to see all available colors.")
        sys.exit(1)
    
    # Validate base color
    base_color_name = validate_single_color(args.color)[1]
    if not base_color_name:
        print(f"Error: Color '{args.color}' not found.")
        print("Use --list-colors to see all available colors.")
        sys.exit(1)
    
    # Get selected tones
    selected_tones = select_tones_for_color(base_color_name, args.tones)
    if not selected_tones:
        print(f"Error: No tones available for {base_color_name}")
        sys.exit(1)
    
    # Find and validate image files
    if args.images:
        selected_images = validate_image_list(args.images, args.images_dir)
    else:
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
    print(f"  Base Color: {base_color_name}")
    print(f"  Tones: {', '.join(selected_tones)}")
    print(f"  Images: {', '.join(os.path.basename(img) for img in valid_images)}")
    print(f"  Results will be saved to: {args.results_dir}")
    
    # Create results directory if it doesn't exist
    os.makedirs(args.results_dir, exist_ok=True)
    
    # Process images
    results = process_images_with_tones(
        valid_images, image_to_mask, base_color_name, selected_tones, args.results_dir
    )
    
    # Visualize results if requested
    if not args.no_visualization and results:
        print("\nVisualizing results...")
        Visualizer.visualize_preview_results(results, args.images_dir)
    
    print("\nDone!")


if __name__ == "__main__":
    main()
