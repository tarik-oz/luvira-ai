"""
Color Changer Package

Advanced hair color change utilities for image segmentation tasks.
Implements HSV-based color transformation and natural blending to achieve realistic hair recoloring.
Designed to work with segmentation masks and user-specified RGB colors.
Suitable for research, prototyping, and integration into larger pipelines.

v3.0.0 updates:
- Per-color special handlers and config for nuanced rendering
- Tone configurations extended for natural variations
"""

__version__ = "3.0.0"
__author__ = "Tarik"
__description__ = (
    "Advanced hair color change utilities with per-color handlers and tone configs."
)

from .core.color_transformer import ColorTransformer
from .config.color_config import COLORS
from .utils.image_utils import ImageUtils
from .utils.color_utils import ColorUtils
from .utils.visualization import Visualizer

__all__ = [
    "ColorTransformer",
    "COLORS",
    "ImageUtils",
    "ColorUtils",
    "Visualizer"
] 