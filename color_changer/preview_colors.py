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
import torch
from pathlib import Path

# Import directly from local files
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import core modules directly
from color_changer.core.color_transformer import ColorTransformer
from color_changer.config.color_config import (
    COLORS, PREVIEW_IMAGES_DIR, PREVIEW_RESULTS_DIR, DEFAULT_MODEL_PATH
)

# Create local imports for runner and visualizer
from color_changer.utils.preview_runner import PreviewRunner
from color_changer.utils.visualization import Visualizer

# Import model predictor for automatic mask generation
from model.inference.predict import load_model
from model.inference.predictor import create_predictor


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


def generate_hair_masks(image_paths, model_path, device='auto', results_dir='test_results'):
    """
    Generate hair masks for the given images using the segmentation model.
    
    Args:
        image_paths: List of image paths
        model_path: Path to the trained model
        device: Device to use for inference
        results_dir: Directory to save mask results
        
    Returns:
        Dictionary mapping image paths to mask paths
    """
    print(f"Loading hair segmentation model from {model_path}...")
    try:
        # Load model
        model = load_model(model_path)
        
        # Create predictor
        predictor = create_predictor(model, device=device)
        
        # Generate masks for each image
        image_to_mask = {}
        for img_path in image_paths:
            img_name = os.path.basename(img_path)
            base_name = os.path.splitext(img_name)[0]
            
            print(f"Generating mask for {img_name}...")
            success = predictor.predict_and_save(img_path, base_name, show_visualization=False)
            
            if success:
                # The predictor saves masks to model/test_results directory
                model_results_dir = os.path.join('..', 'model', 'test_results')
                prob_mask_path = os.path.join(model_results_dir, f"{base_name}_prob_mask.png")
                if os.path.exists(prob_mask_path):
                    image_to_mask[img_path] = prob_mask_path
                    print(f"  Using mask from {prob_mask_path}")
            else:
                print(f"  Failed to generate mask for {img_name}")
        
        return image_to_mask
        
    except Exception as e:
        print(f"Error loading model or generating masks: {e}")
        return {}


def find_existing_masks(image_paths, images_dir):
    """
    Find existing masks for the given images.
    
    Args:
        image_paths: List of image paths
        images_dir: Directory containing images and masks
        
    Returns:
        Dictionary mapping image paths to mask paths
    """
    image_to_mask = {}
    model_results_dir = os.path.join('..', 'model', 'test_results')
    
    for img_path in image_paths:
        img_name = os.path.basename(img_path)
        base_name = os.path.splitext(img_name)[0]
        
        # Try to find mask in model/test_results directory
        prob_mask_path = os.path.join(model_results_dir, f"{base_name}_prob_mask.png")
        if os.path.exists(prob_mask_path):
            image_to_mask[img_path] = prob_mask_path
            continue
            
        # If not found in model/test_results, try other locations
        mask_patterns = [
            f"{base_name}_prob_mask.png",
            f"{base_name}_mask.png",
            f"{base_name}_segmentation.png"
        ]
        
        found_mask = False
        for pattern in mask_patterns:
            mask_path = os.path.join(images_dir, pattern)
            if os.path.exists(mask_path):
                image_to_mask[img_path] = mask_path
                found_mask = True
                break
        
        if not found_mask and img_path not in image_to_mask:
            print(f"Warning: No mask found for {img_name}")
    
    return image_to_mask


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
        selected_images = []
        for img_name in args.images:
            img_path = os.path.join(args.images_dir, img_name)
            if os.path.exists(img_path):
                selected_images.append(img_path)
            else:
                print(f"Warning: Image '{img_name}' not found, ignoring.")
    else:
        # Find all image files in the directory
        image_files = [f for f in os.listdir(args.images_dir) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png')) and not f.endswith('_mask.png') and not 'mask' in f]
        selected_images = [os.path.join(args.images_dir, f) for f in image_files]
    
    if not selected_images:
        print(f"No images found in {args.images_dir}")
        sys.exit(1)
    
    # Get masks for the selected images
    if args.use_existing_masks:
        # Use existing masks
        image_to_mask = find_existing_masks(selected_images, args.images_dir)
    else:
        # Generate masks using the model
        image_to_mask = generate_hair_masks(
            selected_images, 
            args.model, 
            device=args.device,
            results_dir=args.results_dir
        )
    
    # Filter out images without masks
    valid_images = [img for img in selected_images if img in image_to_mask]
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
        image = cv2.imread(img_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
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
                cv2.imwrite(out_path, cv2.cvtColor(result, cv2.COLOR_RGB2BGR))
                
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