"""
Dependencies for Hair Segmentation API
"""

from fastapi import Depends
from ..services import ModelService, PredictionService, ColorChangeService

# Service instances (singleton pattern)
_model_service: ModelService = None
_prediction_service: PredictionService = None
_color_change_service: ColorChangeService = None


def get_model_service() -> ModelService:
    """Get model service instance (singleton)"""
    global _model_service
    if _model_service is None:
        _model_service = ModelService()
    return _model_service


def get_prediction_service() -> PredictionService:
    """Get prediction service instance (singleton)"""
    global _prediction_service
    if _prediction_service is None:
        _prediction_service = PredictionService(get_model_service())
    return _prediction_service


def get_color_change_service() -> ColorChangeService:
    """Get color change service instance (singleton)"""
    global _color_change_service
    if _color_change_service is None:
        _color_change_service = ColorChangeService(get_prediction_service())
    return _color_change_service 