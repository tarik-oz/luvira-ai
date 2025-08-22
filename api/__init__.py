"""
API package for Hair Segmentation

Modular FastAPI REST API for hair segmentation and color change workflows.
Provides endpoints for mask prediction, hair color change, model management, and health checks.
Designed for easy integration with machine learning pipelines and frontend applications.
Includes robust validation, error handling, and extensible service structure.

v3.0.0 updates:
- Frontend-oriented endpoints added
- Session-based workflow implemented
"""

try:
    from .. import __version__, __author__, __description__
except ImportError:
    # Fallback for when running as top-level module
    __version__ = "3.0.0"
    __author__ = "Tarik"
    __description__ = (
        "FastAPI service with frontend endpoints and session-based workflow for hair segmentation."
    )

from .schemas import *
__all__ = ["schemas", "__version__"] 