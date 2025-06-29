"""
FastAPI application for hair segmentation
"""

import tempfile
from pathlib import Path

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import Response
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np

from .model_manager import model_manager
from config import API_CONFIG
from .. import __version__

app = FastAPI(
    title="Hair Segmentation API",
    description="API for hair segmentation using UNet model",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=API_CONFIG["cors_origins"],
    allow_credentials=True,
    allow_methods=API_CONFIG["cors_methods"],
    allow_headers=API_CONFIG["cors_headers"],
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

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "message": "Hair Segmentation API is running!",
        "version": __version__,
        "model_loaded": model_manager.is_model_loaded(),
        "endpoints": {
            "docs": "/docs",
            "health": "/health",
            "model_info": "/model-info",
            "reload_model": "/reload-model",
            "predict_mask": "/predict-mask",
            "clear_model": "/clear-model"
        }
    }

@app.get("/health")
async def health_check():
    """Detailed health check with model status"""
    return {
        "status": "healthy",
        "model_loaded": model_manager.is_model_loaded(),
        "model_path": str(model_manager.get_model_path()) if model_manager.get_model_path() else None
    }

@app.get("/model-info")
async def get_model_info():
    """Get information about the loaded model"""
    return model_manager.get_model_info()

@app.post("/reload-model")
async def reload_model(model_path: str = DEFAULT_MODEL_PATH):
    """
    Reload the model from a different path
    
    Args:
        model_path: Path to the model file (optional, defaults to DEFAULT_MODEL_PATH)
        
    Returns:
        Success status and model info
    """
    # Validate model path exists
    model_file = Path(model_path)
    if not model_file.exists():
        raise HTTPException(
            status_code=400,
            detail=f"Model file not found: {model_path}"
        )
    
    success = model_manager.reload_model(model_path)
    if success:
        return {
            "success": True,
            "message": f"Model reloaded successfully from {model_path}",
            "model_info": model_manager.get_model_info()
        }
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
    
    # Validate file type
    if file.content_type not in API_CONFIG["allowed_image_types"]:
        raise HTTPException(
            status_code=400, 
            detail=f"File type not allowed. Allowed types: {API_CONFIG['allowed_image_types']}"
        )
    
    # Validate file size
    if file.size and file.size > API_CONFIG["max_file_size"]:
        raise HTTPException(
            status_code=400,
            detail=f"File too large. Maximum size: {API_CONFIG['max_file_size'] // (1024*1024)}MB"
        )
    
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

@app.post("/clear-model")
async def clear_model():
    """Clear the loaded model from memory"""
    try:
        model_manager.clear_model()
        return {
            "success": True,
            "message": "Model cleared from memory"
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error clearing model: {str(e)}"
        ) 