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
            "clear_model": "/clear-model",
            "predict_mask": "/predict-mask",
            "change_hair_color": "/change-hair-color", 
            "available_colors": "/available-colors",
            "available_tones": "/available-tones/{color_name}",
            "upload_and_prepare": "/upload-and-prepare",
            "change_hair_color_with_session": "/change-hair-color-with-session/{session_id}",
            "overlays_with_session": "/overlays-with-session/{session_id}",
            "session_stats": "/session-stats",
            "cleanup_session": "/cleanup-session/{session_id}",
            "cleanup_all_sessions": "/cleanup-all-sessions"
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