# Color Transformation Module

## 1. Overview

This module handles realistic hair color transformation using HSV-based algorithms, natural blending techniques, and special color effects. It provides the core color changing functionality used by the API and can be used standalone for testing and development.

## 2. Tech Stack

- **OpenCV** - Advanced image processing and color space conversions
- **NumPy** - Numerical operations and array processing
- **Matplotlib** - Color visualization and testing tools
- **scikit-learn** - Machine learning utilities for color analysis

## 🚀 Getting Started (Local Development)

### Prerequisites

- Python 3.10+
- A virtual environment (e.g., `venv`) is recommended

### Installation & Running

1. **Create and activate a virtual environment from the project root:**

   ```bash
   python -m venv venv
   source venv/bin/activate
   ```

2. **Install dependencies from the root requirements.txt:**

   ```bash
   pip install -r requirements.txt
   ```

3. **Test color transformation:**

   ```bash
   # Navigate to color_changer directory
   cd color_changer

   # Preview color on a sample image
   python preview_colors.py --images /path/to/image.jpg --colors red

   # Preview specific color tones
   python preview_tones.py --images /path/to/image.jpg --color Purple --tones Plum
   ```

> **Note**: For detailed command-line arguments and default parameters, see `preview_colors.py` and `preview_tones.py` files directly for all available options.

## 🎯 Key Features

- **HSV Color Space Algorithm**: Advanced color transformation preserving natural hair highlights and shadows
- **Multi-Tone System**: Each base color supports multiple tone variations (golden, ash, platinum, etc.)
- **Special Effects Handling**: Intelligent processing for extreme colors like gray and white transformations
- **Natural Blending**: Edge smoothing and seamless color integration with original hair texture
- **Preview Tools**: Comprehensive CLI tools for testing and visualizing all color/tone combinations

## ⚙️ Advanced Configuration

For detailed color tuning, hair type optimization, and artifact prevention, see the comprehensive configuration guide:

**📚 [Color Configuration Guide](config/README.md)**

This guide covers:

- Tuning parameters for different base hair types (dark vs light)
- Preventing artifacts like pastel look or pixel stepping
- Optimizing hue, saturation, and brightness settings
- Troubleshooting common color transformation issues

---

For more information about the overall project, see the main [README](../README.md).
