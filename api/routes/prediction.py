"""
Prediction routes for Hair Segmentation API
"""

from fastapi import APIRouter, File, UploadFile, HTTPException, Depends, Form
from fastapi.responses import Response
from ..core import get_prediction_service, get_color_change_service
from ..services import PredictionService, ColorChangeService
from ..schemas.dto import ColorChangeRequest

router = APIRouter()


@router.post("/predict-mask")
async def predict_mask(
    file: UploadFile = File(...),
    prediction_service: PredictionService = Depends(get_prediction_service)
):
    """
    Predict hair mask from uploaded image and return as downloadable file
    
    Args:
        file: Image file (jpg, png, etc.)
        
    Returns:
        Mask image file for download
    """
    try:
        # Use prediction service
        mask_bytes = prediction_service.predict_mask_file(file)
        
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


@router.post("/change-hair-color")
async def change_hair_color(
    file: UploadFile = File(...),
    r: int = Form(..., description="Red channel value (0-255)"),
    g: int = Form(..., description="Green channel value (0-255)"),
    b: int = Form(..., description="Blue channel value (0-255)"),
    color_change_service: ColorChangeService = Depends(get_color_change_service)
):
    """
    Change hair color in uploaded image and return as downloadable file
    
    Args:
        file: Image file (jpg, png, etc.)
        r: Red channel value (0-255)
        g: Green channel value (0-255)
        b: Blue channel value (0-255)
        
    Returns:
        Color-changed image file for download
    """
    try:
        # Validate RGB values
        if not (0 <= r <= 255 and 0 <= g <= 255 and 0 <= b <= 255):
            raise HTTPException(
                status_code=400,
                detail="RGB values must be between 0 and 255"
            )
        
        # Create color list
        target_color = [r, g, b]
        
        # Use color change service
        result_bytes = color_change_service.change_hair_color_file(file, target_color)
        
        # Return result as response
        return Response(
            content=result_bytes,
            media_type="image/png",
            headers={"Content-Disposition": f"attachment; filename=color_changed_{file.filename.split('.')[0]}.png"}
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Color change failed: {str(e)}"
        ) 