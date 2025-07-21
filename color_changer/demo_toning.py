#!/usr/bin/env python
"""
Toning Demonstration Script

This script demonstrates the new toning functionality of the color_changer module.
It shows how to:
1. Generate tonal variations of colors
2. Apply toned colors to hair
3. Preview all available tones
4. Get color information

Usage:
    python demo_toning.py --image test.jpg --color red
    python demo_toning.py --image test.jpg --color [255,0,0] --tone vibrant
    python demo_toning.py --show-color-info --color blue
"""

import os
import sys
import argparse
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import cv2

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from color_changer import ColorTransformer, COLORS
from color_changer.utils.color_utils import ColorUtils
from color_changer.utils.image_utils import ImageUtils
from color_changer.config.color_config import CUSTOM_TONES


# For model inference (if needed)
try:
    from model.inference.predict import load_model
    from model.inference.predictor import create_predictor
except ImportError:
    print("Model inference not available. Please provide masks manually.")


def parse_color(color_str):
    """Parse color string to RGB values."""
    # Check if it's a predefined color name
    color_map = {name.lower(): rgb for rgb, name in COLORS}
    
    if color_str.lower() in color_map:
        return color_map[color_str.lower()]
    
    # Try to parse as RGB list
    try:
        if color_str.startswith('[') and color_str.endswith(']'):
            rgb_str = color_str[1:-1]
            rgb = [int(x.strip()) for x in rgb_str.split(',')]
            if len(rgb) == 3 and all(0 <= c <= 255 for c in rgb):
                return rgb
    except:
        pass
    
    raise ValueError(f"Invalid color: {color_str}")


def show_color_info(color):
    """Display comprehensive color information."""
    rgb = parse_color(color)
    print(f"\n=== Color Information for {color} ===")
    print(f"RGB: {rgb}")
    
    # Basic color info
    info = ColorUtils.get_color_info(rgb)
    print(f"Hex: {info['hex']}")
    print(f"HSV: {info['hsv']}")
    print(f"Brightness: {info['brightness']}")
    print(f"Saturation: {info['saturation']}")
    print(f"Temperature: {info['temperature']}")
    
    # Tonal variations
    print(f"\n=== Tonal Variations ===")
    tone_types = ColorUtils.get_available_tones()
    variations = ColorUtils.generate_tonal_variations(rgb, tone_types)
    
    for tone_name, tone_rgb in variations.items():
        tone_config = tone_types[tone_name]
        print(f"{tone_config['name']:8} ({tone_name:7}): RGB{tone_rgb} - {tone_config['description']}")


def demonstrate_toning(image_path, color, tone_type=None, intensity="moderate", mask_path=None):
    """Demonstrate toning functionality on an image."""
    print(f"\n=== Toning Demonstration ===")
    print(f"Image: {image_path}")
    print(f"Base Color: {color}")
    print(f"Tone: {tone_type or 'all tones'}")
    print(f"Intensity: {intensity}")
    # Parse color
    base_rgb = parse_color(color)
    print(f"Base RGB: {base_rgb}")
    # Load image
    if not os.path.exists(image_path):
        print(f"Error: Image not found: {image_path}")
        return
    image = ImageUtils.load_image(image_path)
    if image is None:
        print(f"Error: Could not load image: {image_path}")
        return
    print(f"Loaded image: {image.shape}")
    # Load or generate mask
    mask = None
    if mask_path and os.path.exists(mask_path):
        mask = ImageUtils.load_image(mask_path, grayscale=True)
        print(f"Using provided mask: {mask_path}")
    else:
        image_dir = os.path.dirname(image_path)
        image_filename = os.path.basename(image_path)
        base_name = os.path.splitext(image_filename)[0]
        mask_patterns = [
            os.path.join(image_dir, f"{base_name}_prob_mask.png"),
            os.path.join(image_dir, f"{base_name}_mask.png"),
            os.path.join(image_dir, f"{base_name}.png")
        ]
        mask_found = False
        for mask_pattern in mask_patterns:
            if os.path.exists(mask_pattern):
                mask = ImageUtils.load_image(mask_pattern, grayscale=True)
                print(f"Found and using mask: {mask_pattern}")
                mask_found = True
                break
        if not mask_found:
            print("Warning: No hair mask found! Using dummy mask (center region).")
            print("For real hair color change, provide a hair segmentation mask with --mask parameter")
            print(f"Expected mask file: {os.path.join(image_dir, f'{base_name}_prob_mask.png')}")
            h, w = image.shape[:2]
            mask = np.zeros((h, w), dtype=np.uint8)
            mask[h//4:3*h//4, w//4:3*w//4] = 255
    
    transformer = ColorTransformer()
    
    print("\nGenerating grid preview...")
    original_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) if image.shape[2] == 3 else image
    try:
        normal_result = transformer.change_hair_color(image, mask, base_rgb)
    except Exception as e:
        print(f"Error applying base color: {e}")
        normal_result = None

    color_name = None
    for rgb, name in COLORS:
        if base_rgb == list(rgb):
            color_name = name
            break
    try:
        if color_name in CUSTOM_TONES:
            previews = {}
            for tone_name, params in CUSTOM_TONES[color_name].items():
                toned_rgb = ColorUtils.create_custom_tone(base_rgb, params["saturation_factor"], params["brightness_factor"], intensity=1.0)
                result = transformer.change_hair_color(image, mask, toned_rgb)
                previews[tone_name] = result
        else:
            previews = transformer.generate_tone_previews(image, mask, base_rgb, intensity)
    except Exception as e:
        print(f"Error generating tone previews: {e}")
        previews = {}
    n_tones = len(previews)
    fig, axes = plt.subplots(1, 2 + n_tones, figsize=(3 * (2 + n_tones), 3))
    axes[0].imshow(original_rgb)
    axes[0].set_title("Original")
    axes[0].axis("off")
    if normal_result is not None:
        axes[1].imshow(normal_result)
        axes[1].set_title("Selected Color")
        axes[1].axis("off")
    else:
        axes[1].text(0.5, 0.5, "Error", ha='center', va='center')
        axes[1].set_title("Selected Color")
        axes[1].axis("off")
    for idx, (tone_name, result) in enumerate(previews.items()):
        if result is not None:
            axes[2 + idx].imshow(result)
            axes[2 + idx].set_title(tone_name)
            axes[2 + idx].axis("off")
        else:
            axes[2 + idx].text(0.5, 0.5, "Error", ha='center', va='center')
            axes[2 + idx].set_title(tone_name)
            axes[2 + idx].axis("off")
    plt.tight_layout()
    plt.show()
    
    # Initialize transformer
    transformer = ColorTransformer()
    
    if tone_type:
        # Apply specific tone
        print(f"\nApplying tone: {tone_type}")
        
        try:
            result = transformer.change_hair_color_with_tone(
                image, mask, base_rgb, tone_type, intensity
            )
            
            # Save result
            output_path = f"demo_result_{color}_{tone_type}_{intensity}.jpg"
            ImageUtils.save_image(result, output_path, convert_to_bgr=True)
            print(f"Result saved: {output_path}")
            
            # Show tone info
            tone_info = ColorUtils.get_tone_info(base_rgb, tone_type)
            print(f"\nTone Information:")
            print(f"  Type: {tone_info['tone_name']}")
            print(f"  Description: {tone_info['description']}")
            print(f"  Original RGB: {tone_info['base_color']['rgb']}")
            print(f"  Toned RGB: {tone_info['toned_color']['rgb']}")
            print(f"  Saturation Factor: {tone_info['adjustments']['saturation_factor']}")
            print(f"  Brightness Factor: {tone_info['adjustments']['brightness_factor']}")
            
        except Exception as e:
            print(f"Error applying tone: {e}")
    else:
        # Generate all tone previews
        print(f"\nGenerating all tone previews...")
        
        try:
            previews = transformer.generate_tone_previews(image, mask, base_rgb, intensity)
            
            print(f"Generated {len(previews)} tone previews:")
            for tone_name, result in previews.items():
                if result is not None:
                    # output_path = f"demo_preview_{color}_{tone_name}_{intensity}.jpg"
                    # ImageUtils.save_image(result, output_path, convert_to_bgr=True)
                    tone_config = ColorUtils.get_available_tones()[tone_name]
                    print(f"  {tone_config['name']:8}: {output_path}")
                else:
                    print(f"  {tone_name:8}: Failed")
                    
        except Exception as e:
            print(f"Error generating previews: {e}")


def list_available_options():
    """List all available colors, tones, and intensities."""
    print("\n=== Available Colors ===")
    for rgb, name in COLORS:
        print(f"{name:8}: RGB{rgb}")
    
    print(f"\n=== Available Tones ===")
    tones = ColorUtils.get_available_tones()
    for tone_key, tone_config in tones.items():
        print(f"{tone_config['name']:8} ({tone_key:7}): {tone_config['description']}")
    
    print(f"\n=== Available Intensities ===")
    intensities = ColorUtils.get_available_intensities()
    for intensity_key, intensity_value in intensities.items():
        print(f"{intensity_key:8}: {intensity_value}")


def main():
    parser = argparse.ArgumentParser(description='Demonstrate hair color toning functionality')
    
    parser.add_argument('--image', type=str, 
                        help='Path to image file')
    
    parser.add_argument('--color', type=str,
                        help='Color name (e.g., "red") or RGB values (e.g., "[255,0,0]")')
    
    parser.add_argument('--tone', type=str, choices=list(ColorUtils.get_available_tones().keys()),
                        help='Specific tone to apply (default: show all tones)')
    
    parser.add_argument('--intensity', type=str, choices=list(ColorUtils.get_available_intensities().keys()),
                        default='moderate', help='Intensity level (default: moderate)')
    
    parser.add_argument('--mask', type=str,
                        help='Path to hair mask (optional)')
    
    parser.add_argument('--show-color-info', action='store_true',
                        help='Show detailed color information')
    
    parser.add_argument('--list-options', action='store_true',
                        help='List all available options')
    
    args = parser.parse_args()
    
    if args.list_options:
        list_available_options()
        return
    
    if not args.color:
        print("Error: --color is required (except for --list-options)")
        print("Use --list-options to see all available options")
        return
    
    if args.show_color_info:
        show_color_info(args.color)
        if not args.image:
            return
    
    if args.image:
        demonstrate_toning(args.image, args.color, args.tone, args.intensity, args.mask)
    else:
        print("Error: --image is required for toning demonstration")
        print("Use --show-color-info to see color information only")
        print("Use --list-options to see all available options")


if __name__ == "__main__":
    main() 