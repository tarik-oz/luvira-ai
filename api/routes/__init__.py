"""
Routes for Hair Segmentation API
"""

from .health import router as health_router
from .model import router as model_router
from .prediction import router as prediction_router
from .public import router as public_router
from .session import router as session_router

__all__ = [
    "health_router",
    "model_router", 
    "prediction_router",
    "public_router",
    "session_router"
]
