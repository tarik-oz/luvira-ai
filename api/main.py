"""
FastAPI application for hair segmentation
"""

import tempfile
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import Response, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
import base64

from .model_manager import model_manager
from .dto import (
    HealthCheckResponse, ModelInfoResponse, ReloadModelRequest, ReloadModelResponse,
    PredictMaskResponse, ClearModelResponse, RootResponse, ErrorResponse,
    FileUploadValidator, HealthStatus
)
from .config import API_CONFIG
from . import __version__

app = FastAPI(
    title="Hair Segmentation API",
    description="API for hair segmentation using deep learning model",
    version="1.1.0"
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
    try:
        success = model_manager.load_model(DEFAULT_MODEL_PATH)
        if not success:
            print("Warning: Could not load model. API will not work properly.")
        else:
            print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")

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
            "predict_mask_json": "/predict-mask-json",
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
    
    # Validate model path exists
    model_file = Path(model_path)
    if not model_file.exists():
        raise HTTPException(
            status_code=400,
            detail=f"Model file not found: {model_path}"
        )
    
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
    # Check if model is loaded
    if not model_manager.is_model_loaded():
        raise HTTPException(
            status_code=500, 
            detail="Model not loaded."
        )
    
    # Validate file using DTO validator
    validation_error = FileUploadValidator.get_validation_error(file.content_type, file.size)
    if validation_error:
        raise HTTPException(status_code=400, detail=validation_error)
    
    temp_input_path = None
    
    try:
        # Read image file
        image_data = await file.read()
        
        # Convert to numpy array
        nparr = np.frombuffer(image_data, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise HTTPException(
                status_code=400, 
                detail="Could not decode image"
            )
        
        # Create temporary input file
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as temp_input:
            temp_input_path = temp_input.name
        
        # Save temporary image for prediction
        cv2.imwrite(temp_input_path, image)
        
        # Get predictor and make prediction
        predictor = model_manager.get_predictor()
        original_image, predicted_mask, binary_mask = predictor.predict(temp_input_path)
        
        if predicted_mask is None:
            raise HTTPException(
                status_code=500, 
                detail="Failed to generate mask"
            )
        
        # Save mask to file
        mask_255 = (predicted_mask * 255).astype(np.uint8)
        
        # Encode mask as PNG bytes
        _, buffer = cv2.imencode('.png', mask_255)
        mask_bytes = buffer.tobytes()
        
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
    finally:
        # Clean up only input temp file
        if temp_input_path:
            try:
                Path(temp_input_path).unlink(missing_ok=True)
            except Exception:
                pass  # Ignore cleanup errors

@app.post("/predict-mask-json", response_model=PredictMaskResponse)
async def predict_mask_json(file: UploadFile = File(...)):
    """
    Predict hair mask from uploaded image and return as JSON with base64 encoded mask
    
    Args:
        file: Image file (jpg, png, etc.)
        
    Returns:
        PredictMaskResponse with base64 encoded mask and metadata
    """
    # Check if model is loaded
    if not model_manager.is_model_loaded():
        raise HTTPException(
            status_code=500, 
            detail="Model not loaded."
        )
    
    # Validate file using DTO validator
    validation_error = FileUploadValidator.get_validation_error(file.content_type, file.size)
    if validation_error:
        raise HTTPException(status_code=400, detail=validation_error)
    
    temp_input_path = None
    
    try:
        # Read image file
        image_data = await file.read()
        
        # Convert to numpy array
        nparr = np.frombuffer(image_data, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise HTTPException(
                status_code=400, 
                detail="Could not decode image"
            )
        
        # Create temporary input file
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as temp_input:
            temp_input_path = temp_input.name
        
        # Save temporary image for prediction
        cv2.imwrite(temp_input_path, image)
        
        # Get predictor and make prediction
        predictor = model_manager.get_predictor()
        original_image, predicted_mask, binary_mask = predictor.predict(temp_input_path)
        
        if predicted_mask is None:
            raise HTTPException(
                status_code=500, 
                detail="Failed to generate mask"
            )
        
        # Convert mask to base64
        mask_255 = (predicted_mask * 255).astype(np.uint8)
        _, buffer = cv2.imencode('.png', mask_255)
        mask_base64 = base64.b64encode(buffer).decode('utf-8')
        
        # Return DTO response
        return PredictMaskResponse(
            success=True,
            mask_base64=f"data:image/png;base64,{mask_base64}",
            original_filename=file.filename,
            mask_shape=list(predicted_mask.shape),
            confidence_score=float(np.mean(predicted_mask))
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Prediction failed: {str(e)}"
        )
    finally:
        # Clean up only input temp file
        if temp_input_path:
            try:
                Path(temp_input_path).unlink(missing_ok=True)
            except Exception:
                pass  # Ignore cleanup errors

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