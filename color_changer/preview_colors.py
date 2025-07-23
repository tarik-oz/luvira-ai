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
from color_changer.config.color_config import (
    PREVIEW_IMAGES_DIR, PREVIEW_RESULTS_DIR, DEFAULT_MODEL_PATH
)

# Create local imports for runner and visualizer
from color_changer.utils.visualization import Visualizer
from color_changer.utils.preview_utils import (
    find_image_files, validate_image_list, get_valid_images_with_masks,
    handle_list_commands, select_colors, process_images_with_colors
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
    
    # Handle list commands
    if handle_list_commands(args):
        sys.exit(0)
    
    selected_colors = select_colors(args)
    
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
    print(f"\nApplying {len(selected_colors)} colors on {len(valid_images)} images:")
    print(f"  Colors: {', '.join(name for _, name in selected_colors)}")
    print(f"  Images: {', '.join(os.path.basename(img) for img in valid_images)}")
    print(f"  Results will be saved to: {args.results_dir}")
    
    # Create results directory if it doesn't exist
    os.makedirs(args.results_dir, exist_ok=True)
    
    # Process images
    results = process_images_with_colors(valid_images, image_to_mask, selected_colors, args.results_dir)
    
    # Visualize results if requested
    if not args.no_visualization and results:
        print("\nVisualizing results...")
        Visualizer.visualize_preview_results(results)
    
    print("\nDone!")

if __name__ == "__main__":
    main() 