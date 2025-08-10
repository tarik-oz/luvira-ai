"""
FastAPI application for hair segmentation
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .routes import health_router, model_router, prediction_router, public_router, session_router
from .core import get_model_service, LoggingMiddleware
from .config import API_CONFIG
from .config.logging_config import setup_logging
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
    allow_origins=API_CONFIG["cors_origins"],
    allow_methods=API_CONFIG["cors_methods"],
    allow_headers=API_CONFIG["cors_headers"],
    allow_credentials=True
)

# Include routers
app.include_router(health_router, tags=["Health"])
app.include_router(model_router, tags=["Model"])
app.include_router(prediction_router, tags=["Prediction"])
app.include_router(public_router, tags=["Public"])
app.include_router(session_router, tags=["Session"])

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
