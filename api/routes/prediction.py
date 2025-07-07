"""
Prediction routes for Hair Segmentation API
"""

from fastapi import APIRouter, File, UploadFile, HTTPException, Depends
from fastapi.responses import Response
from ..core import get_prediction_service
from ..services import PredictionService

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