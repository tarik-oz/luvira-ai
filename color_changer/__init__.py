"""
Color Changer Package

Advanced hair color change utilities for image segmentation tasks.
Implements HSV-based color transformation and natural blending to achieve realistic hair recoloring.
Designed to work with segmentation masks and user-specified RGB colors.
Suitable for research, prototyping, and integration into larger pipelines.
"""

try:
    from .. import __version__, __author__, __description__
except ImportError:
    # Fallback for when running as top-level module
    __version__ = "2.0.0"
    __author__ = "Tarik"
    __description__ = "Advanced hair color change utilities for image segmentation tasks."

from .color_changer import HairColorChanger

__all__ = ["HairColorChanger"] 