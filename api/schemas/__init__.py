"""
Schemas for Hair Segmentation API
"""

from .dto import *

__all__ = [
    "HealthStatus",
    "HealthCheckResponse",
    "ModelInfoResponse",
    "ReloadModelRequest",
    "ReloadModelResponse",
    "ClearModelResponse",
    "RootResponse",
    "ColorChangeRequest"
] 