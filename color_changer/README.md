# Hair Color Changer Module

A comprehensive module for realistic hair color transformations in images using advanced HSV-based color processing and natural blending techniques.

## Overview

This module provides tools to change the color of hair in portrait images using segmentation masks. It leverages advanced color space transformations, texture preservation algorithms, and specialized handling for different target colors to create realistic results.

## Features

- Advanced HSV color space transformations
- Natural texture and lighting preservation
- Special handling for challenging colors (blue, purple, grey, etc.)
- Automatic parameter adjustment based on hair and target color characteristics
- Comprehensive testing and visualization tools
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
│   ├── color_config.py     # Color definitions
│   └── __init__.py
├── utils/                  # Utility functions
│   ├── color_utils.py      # Color utility functions
│   ├── image_utils.py      # Image processing utilities
│   └── __init__.py
├── test_utils/             # Testing utilities
│   ├── test_runner.py      # Test execution
│   ├── visualization.py    # Result visualization
│   └── __init__.py
├── test_images/            # Sample test images
└── __init__.py             # Package initialization
```

## Usage

### Basic Usage

```python
from color_changer import ColorTransformer
import cv2

# Create transformer
transformer = ColorTransformer()

# Load image and mask
image = cv2.imread("image.jpg")
mask = cv2.imread("hair_mask.png", cv2.IMREAD_GRAYSCALE)

# Define target RGB color (R, G, B format, 0-255)
target_color = [0, 0, 255]  # Blue

# Transform image
result = transformer.change_hair_color(image, mask, target_color)

# Save result (result is in RGB format)
cv2.imwrite("result.png", cv2.cvtColor(result, cv2.COLOR_RGB2BGR))
```

### Running Tests

```python
from color_changer import TestRunner, Visualizer

# Create test runner
runner = TestRunner("test_images", "test_results")

# Run tests with all predefined colors
results = runner.run_batch_test()

# Visualize results
Visualizer.visualize_test_results(results)
```

You can also use the command-line test tool:

```bash
python -m color_changer.run_color_changer --list-colors  # List available colors
python -m color_changer.run_color_changer --colors Blue Red  # Test specific colors
```

## Using Utilities

```python
from color_changer import ImageUtils, ColorUtils

# Image utilities
image = ImageUtils.load_image("image.jpg")
resized = ImageUtils.resize_image(image, max_dimension=600)

# Color utilities
hsv = ColorUtils.rgb_to_hsv([255, 0, 0])  # Convert RGB to HSV
complementary_colors = ColorUtils.generate_complementary_colors([255, 0, 0])
```

## Integration with API

This module is designed to work seamlessly with the hair segmentation API, providing a complete pipeline for hair color changing services.
