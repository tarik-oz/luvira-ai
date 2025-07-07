"""
API package for hair segmentation

This package provides FastAPI endpoints for hair segmentation using trained U-Net models.
Currently in demo version with basic endpoints for segmentation and model management.
"""

__version__ = "1.1.0"

from .schemas import *

__all__ = ["schemas", "__version__"] 