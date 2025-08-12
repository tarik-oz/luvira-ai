"""
Frontend-specific routes for Hair Segmentation API
These endpoints are optimized for frontend applications
"""

from fastapi import APIRouter, File, UploadFile, HTTPException, Depends, Form
from fastapi.responses import Response
from ..core import get_color_change_service, SessionExpiredException
from ..services import ColorChangeService

router = APIRouter()

@router.post("/upload-and-prepare")
async def upload_and_prepare(
    file: UploadFile = File(...),
    color_change_service: ColorChangeService = Depends(get_color_change_service)
):
    """
    Upload image, validate it and prepare for fast color changes
    
    This endpoint is optimized for frontend applications that need
    to upload an image once and then perform multiple color changes.
    
    Args:
        file: Image file (jpg, png, etc.)
        
    Returns:
        Session ID for subsequent fast operations
    """
    try:
        session_id = color_change_service.upload_and_prepare_image(file)
        
        return {
            "session_id": session_id,
            "message": "Image uploaded and prepared successfully",
            "expires_in_minutes": 30
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Image upload failed: {str(e)}"
        )

@router.post("/overlays-with-session/{session_id}")
async def overlays_with_session(
    session_id: str,
    color_name: str = Form(..., description="Hair color name (e.g., Blonde, Brown, etc.)"),
    color_change_service: ColorChangeService = Depends(get_color_change_service)
):
    """
    Return a ZIP containing WEBP hair overlays for base color and all tones using cached session data.

    Files inside ZIP:
      - base.webp
      - tones/{tone}.webp
      - metadata.json
    """
    try:
        zip_bytes = color_change_service.build_overlays_with_all_tones_session(session_id, color_name)
        filename = f"overlays_{color_name.lower()}_{session_id}.zip"
        return Response(
            content=zip_bytes,
            media_type="application/zip",
            headers={"Content-Disposition": f"attachment; filename={filename}"}
        )
    except SessionExpiredException as e:
        raise HTTPException(status_code=e.status_code, detail=e.detail)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Overlay bundle generation failed: {str(e)}")