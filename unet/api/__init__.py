"""
API package for UNet hair segmentation

This package provides FastAPI endpoints for hair segmentation using trained U-Net models.
Currently in demo version with basic endpoints for segmentation and model management.
"""

from .. import __version__, __author__

from .model_manager import model_manager

__all__ = ["model_manager"] 