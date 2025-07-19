# Hair Color Changer Module

A comprehensive module for realistic hair color transformations in images using advanced HSV-based color processing and natural blending techniques.

## Overview

This module provides tools to change the color of hair in portrait images using segmentation masks. It leverages advanced color space transformations, texture preservation algorithms, and specialized handling for different target colors to create realistic results. The module now includes automatic hair mask generation.

## Features

- Advanced HSV color space transformations
- Natural texture and lighting preservation
- Special handling for challenging colors (blue, purple, grey, etc.)
- Automatic hair mask generation using segmentation models
- Automatic parameter adjustment based on hair and target color characteristics
- Comprehensive CLI tools for easy testing and preview
- Clean, modular architecture

## Module Structure

```
color_changer/
├── core/                   # Core functionality
│   ├── color_transformer.py   # Main transformation class
│   └── __init__.py
├── transformers/           # Color transformation components
│   ├── blender.py          # Natural blending
│   ├── hsv_transformer.py  # HSV transformations
│   ├── special_color_handler.py  # Special color handling
│   └── __init__.py
├── config/                 # Configuration
│   ├── color_config.py     # Color definitions and default paths
│   └── __init__.py
├── utils/                  # Utility functions
│   ├── color_utils.py      # Color utility functions
│   ├── image_utils.py      # Image processing utilities
│   ├── preview_runner.py   # Preview execution utilities
│   ├── visualization.py    # Preview result comparison grids
│   └── __init__.py
├── test_images/            # Sample test images
├── preview_colors.py       # Main CLI tool for color preview
└── __init__.py             # Package initialization
```

## Usage

### Command Line Interface

The module provides convenient CLI tools for testing and previewing hair color changes:

#### Main Preview Tool

```bash
# Preview all available colors
python preview_colors.py --image test.jpg

# Preview specific colors
python preview_colors.py --image test.jpg --colors red blue purple

# Preview with specific
python preview_colors.py --image test.jpg --color red

# Preview with custom mask
python preview_colors.py --image test.jpg --mask hair_mask.png --color blue

# List all available colors
python preview_colors.py --list-colors
```

### Programmatic Usage

#### Basic Usage with Automatic Mask Generation

```python
from color_changer import ColorTransformer
import cv2

# Create transformer (automatically loads segmentation model)
transformer = ColorTransformer()

# Load image
image = cv2.imread("image.jpg")

# Transform with automatic mask generation
result = transformer.change_hair_color_auto(image, "red")

# Save result
cv2.imwrite("result.png", cv2.cvtColor(result, cv2.COLOR_RGB2BGR))
```

#### Advanced Usage with Custom Mask

```python
from color_changer import ColorTransformer
import cv2

# Create transformer
transformer = ColorTransformer()

# Load image and mask
image = cv2.imread("image.jpg")
mask = cv2.imread("hair_mask.png", cv2.IMREAD_GRAYSCALE)

# Define target color
target_color = "blue"

# Transform image
result = transformer.change_hair_color(image, mask, target_color)

# Save result
cv2.imwrite("result.png", cv2.cvtColor(result, cv2.COLOR_RGB2BGR))
```

#### Using Color Names

```python
from color_changer import ColorTransformer

transformer = ColorTransformer()

# Available colors: red, orange, yellow, green, blue, purple, pink, grey, brown, black, white

# Examples
result1 = transformer.change_hair_color_auto(image, "red")
result2 = transformer.change_hair_color_auto(image, "blue")
result3 = transformer.change_hair_color_auto(image, "purple")
```

### Using Utilities

```python
from color_changer.utils import ImageUtils, ColorUtils
from color_changer.utils.preview_runner import PreviewRunner

# Image utilities
image = ImageUtils.load_image("image.jpg")
resized = ImageUtils.resize_image(image, max_dimension=600)

# Color utilities
hsv = ColorUtils.rgb_to_hsv([255, 0, 0])  # Convert RGB to HSV
complementary_colors = ColorUtils.generate_complementary_colors([255, 0, 0])

# Preview runner for batch processing
runner = PreviewRunner()
results = runner.run_preview(image, ["red", "blue", "purple"])
```

## Configuration

The module uses centralized configuration in `config/color_config.py`:

- **Default paths**: Model paths, image directories, results directories
- **Color definitions**: Predefined colors with HSV values
- **Model settings**: Segmentation model configuration

## Integration with API

This module is designed to work seamlessly with the hair segmentation API, providing a complete pipeline for hair color changing services. The automatic mask generation feature integrates directly with the trained U-Net segmentation model.

## Examples

### Basic Color Change

```bash
python preview_colors.py --image portrait.jpg --color red
```

### Multiple Colors

```bash
python preview_colors.py --image portrait.jpg --colors red blue purple
```

### Custom Mask Usage

```bash
python preview_colors.py --image portrait.jpg --mask custom_mask.png --color blue
```

The module automatically handles:

- Image loading and validation
- Mask generation (if not provided)
- Color transformation
- Result visualization and saving
- Error handling and user feedback
