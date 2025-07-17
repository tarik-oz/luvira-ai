#!/usr/bin/env python
"""
Hair color change preview script.

This script provides a command-line interface for previewing the hair color change functionality.
It allows selecting specific images and colors to apply, and visualizes the results.
"""

import os
import sys
import argparse
import matplotlib.pyplot as plt
import cv2
import numpy as np

# Import directly from local files
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import core modules directly
from color_changer.core.color_transformer import ColorTransformer
from color_changer.config.color_config import COLORS

# Create local imports for runner and visualizer
from color_changer.utils.preview_runner import PreviewRunner
from color_changer.utils.visualization import Visualizer


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Preview hair color change functionality.')
    
    parser.add_argument('--images-dir', default='test_images',
                        help='Directory containing images (default: test_images)')
    
    parser.add_argument('--results-dir', default='test_results',
                        help='Directory to save results (default: test_results)')
    
    parser.add_argument('--colors', nargs='+', default=[],
                        help='Names of colors to apply (default: all colors)')
    
    parser.add_argument('--images', nargs='+', default=[],
                        help='Specific image files to process (default: all images in directory)')
    
    parser.add_argument('--no-visualization', action='store_true',
                        help='Disable visualization of results')
    
    parser.add_argument('--list-colors', action='store_true',
                        help='List available color names and exit')
    
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
    
    # Create preview runner
    preview_runner = PreviewRunner(args.images_dir, args.results_dir)
    
    # Find images
    all_images = preview_runner.find_preview_images()
    if not all_images:
        print(f"No images found in {args.images_dir}")
        sys.exit(1)
    
    # Filter images if specified
    if args.images:
        selected_images = [img for img in all_images if img in args.images]
        if not selected_images:
            print(f"No matching images found for {args.images}")
            sys.exit(1)
    else:
        selected_images = all_images
    
    # Print preview setup
    print(f"Applying {len(selected_colors)} colors on {len(selected_images)} images:")
    print(f"  Colors: {', '.join(name for _, name in selected_colors)}")
    print(f"  Images: {', '.join(selected_images)}")
    print(f"  Results will be saved to: {args.results_dir}")
    
    # Run batch preview
    results = []
    for image in selected_images:
        # Process each image
        image_results = preview_runner.process_image(image, selected_colors)
        if image_results:
            results.append((image, image_results))
    
    # Visualize results if requested
    if not args.no_visualization and results:
        print("\nVisualizing results...")
        Visualizer.visualize_preview_results(results)
    
    print("\nDone!")


if __name__ == "__main__":
    main() 