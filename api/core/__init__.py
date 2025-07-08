"""
Core modules for Hair Segmentation API
"""

from .exceptions import (
    APIException,
    ModelNotLoadedException,
    ModelLoadException,
    FileValidationException,
    PredictionException,
    ImageProcessingException,
    create_error_response
)
from .dependencies import get_model_service, get_prediction_service, get_color_change_service
from .middleware import LoggingMiddleware, CORSMiddleware

__all__ = [
    "APIException",
    "ModelNotLoadedException", 
    "ModelLoadException",
    "FileValidationException",
    "PredictionException",
    "ImageProcessingException",
    "create_error_response",
    "get_model_service",
    "get_prediction_service",
    "get_color_change_service",
    "LoggingMiddleware",
    "CORSMiddleware"
]
