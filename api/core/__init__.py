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
    SessionExpiredException
)
from .dependencies import get_model_service, get_prediction_service, get_color_change_service
from .middleware import LoggingMiddleware
from .exception_handlers import register_exception_handlers

__all__ = [
    "APIException",
    "ModelNotLoadedException", 
    "ModelLoadException",
    "FileValidationException",
    "PredictionException",
    "ImageProcessingException",
    "SessionExpiredException",
    "get_model_service",
    "get_prediction_service",
    "get_color_change_service",
    "LoggingMiddleware",
    "register_exception_handlers"
]
