# Color Transformation Module

This module handles realistic hair color transformation using HSV-based algorithms, natural blending techniques, and special color effects. It provides the core color changing functionality used by the API and can be used standalone for testing and development.

## 📋 Table of Contents

- [Setup](#setup)
- [Color System Overview](#color-system-overview)
- [Quick Start](#quick-start)
- [CLI Tools](#cli-tools)
- [Configuration](#configuration)
- [Advanced Usage](#advanced-usage)
- [Custom Colors](#custom-colors)
- [Testing Tools](#testing-tools)
- [API Reference](#api-reference)

## 🚀 Setup

### Prerequisites

- Python 3.10+
- Virtual environment (recommended)

### 1. Create Virtual Environment

```bash
# Navigate to project root
cd deep-learning-hair-segmentation

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate
```

### 2. Install Dependencies

```bash
# Install all project dependencies
pip install -r requirements.txt

# Or install color_changer specific dependencies
pip install opencv-python numpy matplotlib scikit-learn
```

### 3. Verify Installation

```bash
# Test color_changer import
python -c "from color_changer import ColorTransformer; print('Color transformer imported successfully')"

# Check available colors
python -c "from color_changer.config.color_config import COLORS; print(f'Available colors: {len(COLORS)}')"
```

## 🎨 Color System Overview

### Architecture

```
color_changer/
├── config/              # Color definitions and settings
│   └── color_config.py  # COLORS, CUSTOM_TONES, SPECIAL_COLORS
├── core/                # Main transformation logic
│   └── color_transformer.py # Primary API interface
├── transformers/        # Specialized transformation algorithms
│   ├── hsv_transformer.py   # HSV color space transformations
│   ├── blender.py          # Natural blending algorithms
│   └── special_color_handler.py # Special effects (gray, white, etc.)
└── utils/               # Helper utilities
    ├── color_utils.py   # Color space conversions
    ├── hsv_utils.py     # HSV manipulation functions
    ├── image_utils.py   # Image processing helpers
    └── visualization.py # Preview and testing tools
```

### Color Transformation Pipeline

```
Original Image + Hair Mask
    ↓
HSV Color Space Conversion
    ↓
Target Color Application
    ↓
Tone Variation (if specified)
    ↓
Natural Blending & Edge Smoothing
    ↓
Special Effects (Gray/White handling)
    ↓
RGB Conversion & Final Output
```

## 🚀 Quick Start

### Basic Color Change

```python
from color_changer import ColorTransformer
import cv2
import numpy as np

# Initialize transformer
transformer = ColorTransformer()

# Load image and mask
image = cv2.imread('original_image.jpg')
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
mask = cv2.imread('hair_mask.png', cv2.IMREAD_GRAYSCALE)

# Change hair color
result = transformer.change_hair_color(image_rgb, mask, "Blonde")

# Save result
result_bgr = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
cv2.imwrite('blonde_hair.jpg', result_bgr)
```

### Color Change with Tone

```python
# Apply specific tone
result = transformer.apply_color_with_tone(image_rgb, mask, "Brown", "golden")

# Available tones vary by color
from color_changer.config.color_config import CUSTOM_TONES
brown_tones = CUSTOM_TONES.get("Brown", {})
print(f"Brown tones: {list(brown_tones.keys())}")
```

## 🛠️ CLI Tools

### Preview All Colors

```bash
# Preview all available colors on a sample image
cd color_changer
python preview_colors.py /path/to/image.jpg /path/to/mask.png

# This creates previews for all colors:
# - preview_colors_Blonde.jpg
# - preview_colors_Brown.jpg
# - preview_colors_Black.jpg
# etc.
```

### Preview Color Tones

```bash
# Preview all tones for a specific color
python preview_tones.py /path/to/image.jpg /path/to/mask.png Brown

# This creates tone previews:
# - preview_tones_Brown_base.jpg
# - preview_tones_Brown_golden.jpg
# - preview_tones_Brown_ash.jpg
# etc.
```

### Test All Combinations

```bash
# Generate comprehensive test matrix
python test_all_colors_tones.py /path/to/image.jpg /path/to/mask.png

# Creates organized output:
# test_results/
# ├── Blonde/
# │   ├── base.jpg
# │   ├── golden.jpg
# │   └── ash.jpg
# ├── Brown/
# │   ├── base.jpg
# │   ├── golden.jpg
# │   └── chocolate.jpg
# └── ...
```

### Quick Color Test

```bash
# Test single color transformation
python -c "
from color_changer import ColorTransformer
import cv2

transformer = ColorTransformer()
image = cv2.imread('image.jpg')
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
mask = cv2.imread('mask.png', cv2.IMREAD_GRAYSCALE)

result = transformer.change_hair_color(image_rgb, mask, 'Red')
result_bgr = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
cv2.imwrite('red_hair_test.jpg', result_bgr)
print('Test completed: red_hair_test.jpg')
"
```

## ⚙️ Configuration

### Available Colors

View and modify colors in `color_changer/config/color_config.py`:

```python
# Base colors (RGB values)
COLORS = [
    ([45, 20, 15], "Black"),
    ([101, 67, 33], "Brown"),
    ([165, 42, 42], "Auburn"),
    ([220, 208, 186], "Blonde"),
    ([255, 0, 0], "Red"),
    ([128, 0, 128], "Purple"),
    ([0, 128, 0], "Green"),
    ([0, 0, 255], "Blue"),
    ([255, 192, 203], "Pink"),
    ([255, 165, 0], "Orange"),
    ([192, 192, 192], "Silver"),
    ([255, 255, 255], "White")
]
```

### Color Tones

Each base color can have multiple tones:

```python
CUSTOM_TONES = {
    "Blonde": {
        "golden": {"hue_shift": 5, "saturation_boost": 0.3},
        "ash": {"hue_shift": -10, "saturation_boost": -0.2},
        "platinum": {"hue_shift": 0, "saturation_boost": -0.6},
        "strawberry": {"hue_shift": 15, "saturation_boost": 0.4}
    },
    "Brown": {
        "golden": {"hue_shift": 8, "saturation_boost": 0.25},
        "ash": {"hue_shift": -15, "saturation_boost": -0.3},
        "chocolate": {"hue_shift": -5, "saturation_boost": 0.2},
        "mahogany": {"hue_shift": 20, "saturation_boost": 0.4}
    }
    # ... more colors
}
```

### Special Effects

Special handling for extreme transformations:

```python
SPECIAL_COLORS = {
    "Gray": {
        "target_rgb": [128, 128, 128],
        "desaturation_factor": 0.7,
        "brightness_adjustment": 0.1
    },
    "White": {
        "target_rgb": [255, 255, 255],
        "desaturation_factor": 0.9,
        "brightness_adjustment": 0.4
    }
}
```

## 🔬 Advanced Usage

### Custom Color Transformation

```python
from color_changer.core.color_transformer import ColorTransformer
from color_changer.transformers.hsv_transformer import HSVTransformer

# Initialize with custom settings
transformer = ColorTransformer()

# Access internal components
hsv_transformer = transformer.hsv_transformer

# Manual HSV transformation
import numpy as np
target_hsv = np.array([30, 180, 200])  # Custom HSV values
result = hsv_transformer.transform_to_target_hsv(image_rgb, mask, target_hsv)
```

### Batch Processing

```python
import os
from pathlib import Path

def batch_color_change(image_dir, mask_dir, output_dir, color_name):
    transformer = ColorTransformer()

    for img_file in Path(image_dir).glob("*.jpg"):
        mask_file = Path(mask_dir) / f"{img_file.stem}.png"

        if mask_file.exists():
            # Load and process
            image = cv2.imread(str(img_file))
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            mask = cv2.imread(str(mask_file), cv2.IMREAD_GRAYSCALE)

            # Transform
            result = transformer.change_hair_color(image_rgb, mask, color_name)

            # Save
            output_file = Path(output_dir) / f"{img_file.stem}_{color_name.lower()}.jpg"
            result_bgr = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(output_file), result_bgr)

            print(f"Processed: {img_file.name} -> {output_file.name}")

# Usage
batch_color_change("input_images/", "input_masks/", "output/", "Blonde")
```

### Fine-tuning Parameters

```python
# Access and modify transformation parameters
from color_changer.transformers.hsv_transformer import HSVTransformer

# Create custom transformer with adjusted parameters
hsv_transformer = HSVTransformer()

# Modify blending parameters (if accessible)
# These would typically be class attributes or config values
```

## 🎯 Custom Colors

### Adding New Base Colors

Edit `color_changer/config/color_config.py`:

```python
# Add new color to COLORS list
COLORS.append(([255, 20, 147], "Hot Pink"))  # RGB values + name

# Add corresponding tones
CUSTOM_TONES["Hot Pink"] = {
    "electric": {"hue_shift": 10, "saturation_boost": 0.5},
    "pastel": {"hue_shift": -5, "saturation_boost": -0.4},
    "neon": {"hue_shift": 15, "saturation_boost": 0.8}
}
```

### Creating Custom Tones

```python
# Define custom tone variations
CUSTOM_TONES["Blue"] = {
    "navy": {"hue_shift": -20, "saturation_boost": 0.2},
    "sky": {"hue_shift": 10, "saturation_boost": -0.3},
    "electric": {"hue_shift": 5, "saturation_boost": 0.6},
    "midnight": {"hue_shift": -15, "saturation_boost": 0.4}
}
```

### RGB to HSV Conversion Helper

```python
from color_changer.utils.color_utils import rgb_to_hsv

# Convert RGB color to HSV for configuration
rgb_color = [255, 105, 180]  # Hot pink
hsv_color = rgb_to_hsv(rgb_color)
print(f"RGB {rgb_color} = HSV {hsv_color}")
```

## 🧪 Testing Tools

### Visual Quality Assessment

```bash
# Generate side-by-side comparisons
python -c "
from color_changer.utils.visualization import create_comparison_grid
import cv2

original = cv2.imread('original.jpg')
transformed = cv2.imread('transformed.jpg')
comparison = create_comparison_grid([original, transformed], ['Original', 'Transformed'])
cv2.imwrite('comparison.jpg', comparison)
"
```

### Color Accuracy Testing

```python
from color_changer.utils.color_utils import analyze_color_distribution

# Analyze color distribution in result
result_colors = analyze_color_distribution(result_image, hair_mask)
print(f"Dominant colors: {result_colors}")
```

### Performance Benchmarking

```python
import time
from color_changer import ColorTransformer

transformer = ColorTransformer()

# Benchmark transformation speed
start_time = time.time()
result = transformer.change_hair_color(image_rgb, mask, "Brown")
end_time = time.time()

print(f"Transformation time: {end_time - start_time:.2f} seconds")
print(f"Image size: {image_rgb.shape}")
```

### Mask Quality Validation

```python
def validate_mask_quality(mask):
    """Check mask quality metrics"""
    total_pixels = mask.size
    hair_pixels = (mask > 0).sum()
    hair_ratio = hair_pixels / total_pixels

    # Check mask properties
    has_hair = hair_ratio > 0.01  # At least 1% hair
    has_edges = cv2.Laplacian(mask, cv2.CV_64F).var() > 100  # Edge detail

    print(f"Hair ratio: {hair_ratio:.3f}")
    print(f"Has sufficient hair: {has_hair}")
    print(f"Has edge detail: {has_edges}")

    return has_hair and has_edges

# Test mask quality
is_good_mask = validate_mask_quality(mask)
```

## 📚 API Reference

### ColorTransformer Class

#### Primary Methods

```python
# Initialize transformer
transformer = ColorTransformer()

# Basic color change
result = transformer.change_hair_color(image_rgb, mask, color_name)
# Parameters:
#   image_rgb: numpy array (H, W, 3) in RGB format
#   mask: numpy array (H, W) with 0-255 values
#   color_name: string from COLORS list
# Returns: numpy array (H, W, 3) in RGB format

# Color change with tone
result = transformer.apply_color_with_tone(image_rgb, mask, color_name, tone_name)
# Additional parameter:
#   tone_name: string from CUSTOM_TONES[color_name]
```

#### Utility Methods

```python
# Get available colors
colors = transformer.get_available_colors()
# Returns: list of color names

# Get available tones for color
tones = transformer.get_available_tones(color_name)
# Returns: list of tone names

# Validate color/tone combination
is_valid = transformer.is_valid_color_tone(color_name, tone_name)
# Returns: boolean
```

### Configuration Access

```python
# Access color definitions
from color_changer.config.color_config import COLORS, CUSTOM_TONES, SPECIAL_COLORS

# Get RGB value for color name
def get_color_rgb(color_name):
    for rgb, name in COLORS:
        if name == color_name:
            return rgb
    return None

rgb_values = get_color_rgb("Blonde")
print(f"Blonde RGB: {rgb_values}")
```

### Utility Functions

```python
from color_changer.utils.color_utils import (
    rgb_to_hsv, hsv_to_rgb,
    adjust_saturation, adjust_brightness
)
from color_changer.utils.hsv_utils import (
    shift_hue, enhance_saturation,
    blend_colors
)
from color_changer.utils.image_utils import (
    apply_mask, blend_images,
    smooth_edges
)
```

## 🔧 Troubleshooting

### Common Issues

#### 1. Poor Color Quality

```python
# Check input image quality
print(f"Image shape: {image.shape}")
print(f"Image dtype: {image.dtype}")
print(f"Image range: {image.min()} - {image.max()}")

# Ensure proper RGB format
if image.shape[2] == 3 and image.max() <= 1.0:
    image = (image * 255).astype(np.uint8)
```

#### 2. Mask Issues

```python
# Validate mask format
print(f"Mask shape: {mask.shape}")
print(f"Mask dtype: {mask.dtype}")
print(f"Mask range: {mask.min()} - {mask.max()}")
print(f"Unique values: {np.unique(mask)}")

# Ensure mask is binary/grayscale
if mask.max() <= 1.0:
    mask = (mask * 255).astype(np.uint8)
```

#### 3. Color Not Available

```python
# Check available colors
from color_changer.config.color_config import COLORS
available_colors = [name for _, name in COLORS]
print(f"Available colors: {available_colors}")

# Check available tones
from color_changer.config.color_config import CUSTOM_TONES
if color_name in CUSTOM_TONES:
    available_tones = list(CUSTOM_TONES[color_name].keys())
    print(f"Available tones for {color_name}: {available_tones}")
```

#### 4. Memory Issues

```python
# Process large images in chunks or resize
def resize_for_processing(image, max_size=1024):
    h, w = image.shape[:2]
    if max(h, w) > max_size:
        scale = max_size / max(h, w)
        new_w, new_h = int(w * scale), int(h * scale)
        return cv2.resize(image, (new_w, new_h))
    return image

# Use smaller image for processing
small_image = resize_for_processing(image_rgb)
small_mask = resize_for_processing(mask)
result_small = transformer.change_hair_color(small_image, small_mask, "Brown")

# Resize result back to original size
result_full = cv2.resize(result_small, (image_rgb.shape[1], image_rgb.shape[0]))
```

### Debug Mode

```python
# Enable detailed logging (if implemented)
import logging
logging.basicConfig(level=logging.DEBUG)

# Manual step-by-step processing
from color_changer.transformers.hsv_transformer import HSVTransformer

transformer = HSVTransformer()
hsv_image = transformer._rgb_to_hsv(image_rgb)
print(f"HSV conversion successful: {hsv_image.shape}")

# Continue with manual processing...
```

### Performance Optimization

```python
# Profile transformation performance
import cProfile
import pstats

def profile_transformation():
    transformer = ColorTransformer()
    return transformer.change_hair_color(image_rgb, mask, "Brown")

# Run profiler
cProfile.run('profile_transformation()', 'color_profile.stats')

# Analyze results
stats = pstats.Stats('color_profile.stats')
stats.sort_stats('cumulative').print_stats(10)
```

---

For more information about the overall project, see the main [README](../README.md).
