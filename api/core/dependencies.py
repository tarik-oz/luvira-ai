"""
Dependencies for Hair Segmentation API
"""

from fastapi import Depends
from ..services import ModelService, PredictionService

# Service instances (singleton pattern)
_model_service: ModelService = None
_prediction_service: PredictionService = None


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