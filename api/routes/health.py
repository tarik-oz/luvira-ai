"""
Health check routes for Hair Segmentation API
"""

from fastapi import APIRouter, Depends
from ..schemas import HealthCheckResponse, RootResponse, HealthStatus
from ..core import get_model_service
from ..services import ModelService
from .. import __version__

router = APIRouter()


@router.get("/", response_model=RootResponse)
async def root(model_service: ModelService = Depends(get_model_service)):
    """Health check endpoint"""
    return RootResponse(
        message="Hair Segmentation API is running!",
        version=__version__,
        model_loaded=model_service.is_model_loaded(),
        endpoints={
            "docs": "/docs",
            "health": "/health",
            "model_info": "/model-info",
            "reload_model": "/reload-model",
            "predict_mask": "/predict-mask",
            "change_hair_color": "/change-hair-color",
            "change_hair_color_rgb": "/change-hair-color-rgb",
            "change_hair_color_all_tones": "/change-hair-color-all-tones",
            "available_colors": "/available-colors",
            "available_tones": "/available-tones/{color_name}",
            "clear_model": "/clear-model"
        }
    )


@router.get("/health", response_model=HealthCheckResponse)
async def health_check(model_service: ModelService = Depends(get_model_service)):
    """Detailed health check with model status"""
    status = HealthStatus.HEALTHY if model_service.is_model_loaded() else HealthStatus.DEGRADED
    
    return HealthCheckResponse(
        status=status,
        model_loaded=model_service.is_model_loaded(),
        model_path=str(model_service.get_model_path()) if model_service.get_model_path() else None,
        version=__version__
    ) 