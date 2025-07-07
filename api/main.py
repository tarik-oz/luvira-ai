"""
FastAPI application for hair segmentation
"""

from typing import Optional

from fastapi import FastAPI
from .core import LoggingMiddleware, CORSMiddleware

from .routes import health_router, model_router, prediction_router
from .core import get_model_service
from .config import API_CONFIG, setup_logging
from . import __version__

# Setup logging
setup_logging()

app = FastAPI(
    title=API_CONFIG["api_title"],
    description=API_CONFIG["api_description"],
    version=API_CONFIG["api_version"]
)

# Add middleware
app.add_middleware(LoggingMiddleware)
app.add_middleware(
    CORSMiddleware,
    origins=API_CONFIG["cors_origins"],
    methods=API_CONFIG["cors_methods"],
    headers=API_CONFIG["cors_headers"]
)

# Include routers
app.include_router(health_router, tags=["Health"])
app.include_router(model_router, tags=["Model"])
app.include_router(prediction_router, tags=["Prediction"])

@app.on_event("startup")
async def startup_event():
    """Load model when API starts"""
    import logging
    logger = logging.getLogger(__name__)
    
    try:
        logger.info("Starting API initialization...")
        model_service = get_model_service()
        success = model_service.load_model()
        if not success:
            logger.warning("Could not load model. API will not work properly.")
        else:
            logger.info("Model loaded successfully!")
    except Exception as e:
        logger.error(f"Error loading model: {e}")
