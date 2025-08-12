"""
Prediction routes for Hair Segmentation API
"""

from fastapi import APIRouter, File, UploadFile, HTTPException, Depends, Form
from fastapi.responses import Response
from typing import Optional
from ..core import get_prediction_service, get_color_change_service, SessionExpiredException
from ..services import PredictionService, ColorChangeService

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


@router.post("/change-hair-color-with-session/{session_id}")
async def change_hair_color_with_session(
    session_id: str,
    color_name: str = Form(..., description="Hair color name (e.g., Blonde, Brown, etc.)"),
    tone: Optional[str] = Form(None, description="Optional tone for the color (e.g., golden, ash, etc.)"),
    color_change_service: ColorChangeService = Depends(get_color_change_service)
):
    """
    Hair color change using cached session data (NO mask generation!)
    
    Args:
        session_id: Session identifier from upload-and-prepare
        color_name: Hair color name from available colors
        tone: Optional tone name for the color
        
    Returns:
        Color-changed image file for download
    """
    try:
        result_bytes = color_change_service.change_hair_color_with_session(session_id, color_name, tone)
        tone_suffix = f"_{tone}" if tone else ""
        filename = f"session_color_{color_name.lower()}{tone_suffix}_{session_id}.png"
        return Response(
            content=result_bytes,
            media_type="image/png",
            headers={"Content-Disposition": f"attachment; filename={filename}"}
        )
    except SessionExpiredException as e:
        raise HTTPException(status_code=e.status_code, detail=e.detail)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Session color change failed: {str(e)}")


@router.get("/available-colors")
async def get_available_colors(
    color_change_service: ColorChangeService = Depends(get_color_change_service)
):
    """
    Get list of available hair colors
    
    Returns:
        List of available color names
    """
    try:
        colors_with_rgb = color_change_service.get_available_colors()
        color_names = [c["name"] for c in colors_with_rgb]
        return {
            "colors": color_names,
            "colors_with_rgb": colors_with_rgb,
            "count": len(color_names)
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get available colors: {str(e)}"
        )


@router.get("/available-tones/{color_name}")
async def get_available_tones(
    color_name: str,
    color_change_service: ColorChangeService = Depends(get_color_change_service)
):
    """
    Get list of available tones for a specific color
    
    Args:
        color_name: Name of the color
        
    Returns:
        List of available tone names for the color
    """
    try:
        tones = color_change_service.get_available_tones(color_name)
        return {
            "color": color_name,
            "tones": tones,
            "count": len(tones)
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get available tones for {color_name}: {str(e)}"
        )
        
@router.post("/change-hair-color")
async def change_hair_color(
    file: UploadFile = File(...),
    color_name: str = Form(..., description="Hair color name (e.g., Blonde, Brown, etc.)"),
    tone: Optional[str] = Form(None, description="Optional tone for the color (e.g., golden, ash, etc.)"),
    color_change_service: ColorChangeService = Depends(get_color_change_service)
):
    """
    Change hair color in uploaded image using color name and optional tone
    
    Args:
        file: Image file (jpg, png, etc.)
        color_name: Hair color name from available colors
        tone: Optional tone name for the color
        
    Returns:
        Color-changed image file for download
    """
    try:
        # Use color change service with the new method
        result_bytes = color_change_service.change_hair_color(file, color_name, tone)
        
        # Create filename
        tone_suffix = f"_{tone}" if tone else ""
        filename = f"color_changed_{color_name.lower()}{tone_suffix}_{file.filename.split('.')[0]}.png"
        
        # Return result as response
        return Response(
            content=result_bytes,
            media_type="image/png",
            headers={"Content-Disposition": f"attachment; filename={filename}"}
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Color change failed: {str(e)}"
        )