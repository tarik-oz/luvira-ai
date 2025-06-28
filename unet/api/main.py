"""
FastAPI application for hair segmentation
"""

from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np

from .model_manager import model_manager

app = FastAPI(
    title="Hair Segmentation API",
    description="API for hair segmentation using UNet model",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DEFAULT_MODEL_PATH = "models/best_model.pth"

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
        "model_loaded": model_manager.is_model_loaded()
    }

@app.get("/model-info")
async def get_model_info():
    """Get information about the loaded model"""
    return model_manager.get_model_info()

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
    if not file.content_type.startswith('image/'):
        raise HTTPException(
            status_code=400, 
            detail="File must be an image"
        )
    
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
        
        # Save temporary image for prediction
        temp_path = "temp_upload.jpg"
        cv2.imwrite(temp_path, image)
        
        # Get predictor and make prediction
        predictor = model_manager.get_predictor()
        original_image, predicted_mask, binary_mask = predictor.predict(temp_path)
        
        if predicted_mask is None:
            raise HTTPException(
                status_code=500, 
                detail="Failed to generate mask"
            )
        
        # Save mask to file
        mask_path = "temp_mask.png"
        mask_255 = (predicted_mask * 255).astype(np.uint8)
        cv2.imwrite(mask_path, mask_255)
        
        # Clean up temp input file
        Path(temp_path).unlink(missing_ok=True)
        
        # Return mask file
        from fastapi.responses import FileResponse
        return FileResponse(
            path=mask_path,
            media_type="image/png",
            filename=f"hair_mask_{file.filename.split('.')[0]}.png"
        )
        
    except Exception as e:
        # Clean up temp files in case of error
        Path("temp_upload.jpg").unlink(missing_ok=True)
        Path("temp_mask.png").unlink(missing_ok=True)
        raise HTTPException(
            status_code=500, 
            detail=f"Prediction failed: {str(e)}"
        ) 