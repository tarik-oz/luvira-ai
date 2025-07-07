"""
FastAPI application for hair segmentation
"""

from typing import Optional

from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import Response, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from .model_manager import model_manager
from .dto import (
    HealthCheckResponse, ModelInfoResponse, ReloadModelRequest, ReloadModelResponse,
    ClearModelResponse, RootResponse, ErrorResponse,
    HealthStatus
)
from .utils import FileValidator, ModelPathValidator
from .services import PredictionService
from .config import API_CONFIG, setup_logging
from . import __version__

# Setup logging
setup_logging()

# Initialize services
prediction_service = PredictionService(model_manager)

app = FastAPI(
    title=API_CONFIG["api_title"],
    description=API_CONFIG["api_description"],
    version=API_CONFIG["api_version"]
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DEFAULT_MODEL_PATH = API_CONFIG["default_model_path"]

@app.on_event("startup")
async def startup_event():
    """Load model when API starts"""
    import logging
    logger = logging.getLogger(__name__)
    
    try:
        logger.info("Starting API initialization...")
        success = model_manager.load_model(DEFAULT_MODEL_PATH)
        if not success:
            logger.warning("Could not load model. API will not work properly.")
        else:
            logger.info("Model loaded successfully!")
    except Exception as e:
        logger.error(f"Error loading model: {e}")

@app.get("/", response_model=RootResponse)
async def root():
    """Health check endpoint"""
    return RootResponse(
        message="Hair Segmentation API is running!",
        version=__version__,
        model_loaded=model_manager.is_model_loaded(),
        endpoints={
            "docs": "/docs",
            "health": "/health",
            "model_info": "/model-info",
            "reload_model": "/reload-model",
            "predict_mask": "/predict-mask",
            "clear_model": "/clear-model"
        }
    )

@app.get("/health", response_model=HealthCheckResponse)
async def health_check():
    """Detailed health check with model status"""
    status = HealthStatus.HEALTHY if model_manager.is_model_loaded() else HealthStatus.DEGRADED
    
    return HealthCheckResponse(
        status=status,
        model_loaded=model_manager.is_model_loaded(),
        model_path=str(model_manager.get_model_path()) if model_manager.get_model_path() else None,
        version=__version__
    )

@app.get("/model-info", response_model=ModelInfoResponse)
async def get_model_info():
    """Get information about the loaded model"""
    model_info = model_manager.get_model_info()
    
    return ModelInfoResponse(
        model_path=model_info.get("model_path"),
        model_type=model_info.get("model_type"),
        input_shape=model_info.get("input_shape"),
        output_shape=model_info.get("output_shape"),
        device=model_info.get("device"),
        is_loaded=model_manager.is_model_loaded()
    )

@app.post("/reload-model", response_model=ReloadModelResponse)
async def reload_model(request: ReloadModelRequest):
    """
    Reload the model from a different path
    
    Args:
        request: ReloadModelRequest containing model path
        
    Returns:
        ReloadModelResponse with success status and model info
    """
    model_path = request.model_path or DEFAULT_MODEL_PATH
    
    # Validate model path using new validator
    ModelPathValidator.validate_model_path(model_path)
    
    success = model_manager.reload_model(model_path)
    if success:
        model_info = model_manager.get_model_info()
        return ReloadModelResponse(
            success=True,
            message=f"Model reloaded successfully from {model_path}",
            model_info=ModelInfoResponse(
                model_path=model_info.get("model_path"),
                model_type=model_info.get("model_type"),
                input_shape=model_info.get("input_shape"),
                output_shape=model_info.get("output_shape"),
                device=model_info.get("device"),
                is_loaded=model_manager.is_model_loaded()
            )
        )
    else:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to reload model from {model_path}"
        )

@app.post("/predict-mask")
async def predict_mask(file: UploadFile = File(...)):
    """
    Predict hair mask from uploaded image and return as downloadable file
    
    Args:
        file: Image file (jpg, png, etc.)
        
    Returns:
        Mask image file for download
    """
    try:
        # Use prediction service
        mask_bytes = await prediction_service.predict_mask_file(file)
        
        # Return mask as response
        return Response(
            content=mask_bytes,
            media_type="image/png",
            headers={"Content-Disposition": f"attachment; filename=hair_mask_{file.filename.split('.')[0]}.png"}
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Prediction failed: {str(e)}"
        )

@app.post("/clear-model", response_model=ClearModelResponse)
async def clear_model():
    """Clear the loaded model from memory"""
    try:
        model_manager.clear_model()
        return ClearModelResponse(
            success=True,
            message="Model cleared from memory"
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error clearing model: {str(e)}"
        ) 