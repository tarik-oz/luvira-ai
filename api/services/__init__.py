"""
Services for Hair Segmentation API
"""

from .prediction_service import PredictionService
from .model_service import ModelService

__all__ = [
    "PredictionService",
    "ModelService"
]
