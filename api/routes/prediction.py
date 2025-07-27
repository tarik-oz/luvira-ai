"""
Prediction routes for Hair Segmentation API
"""

from fastapi import APIRouter, File, UploadFile, HTTPException, Depends, Form
from fastapi.responses import Response
from typing import Optional
import zipfile
import io
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
            "success": True,
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
            "success": True,
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
        result_bytes = color_change_service.change_hair_color_by_name(file, color_name, tone)
        
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


@router.post("/change-hair-color-rgb")
async def change_hair_color_rgb(
    file: UploadFile = File(...),
    r: int = Form(..., description="Red channel value (0-255)"),
    g: int = Form(..., description="Green channel value (0-255)"),
    b: int = Form(..., description="Blue channel value (0-255)"),
    color_change_service: ColorChangeService = Depends(get_color_change_service)
):
    """
    Change hair color in uploaded image using RGB values (legacy method)
    
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
        
        # Use legacy color change service method
        result_bytes = color_change_service.change_hair_color_file(file, target_color)
        
        # Return result as response
        return Response(
            content=result_bytes,
            media_type="image/png",
            headers={"Content-Disposition": f"attachment; filename=color_changed_rgb_{file.filename.split('.')[0]}.png"}
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Color change failed: {str(e)}"
        )


@router.post("/change-hair-color-all-tones")
async def change_hair_color_all_tones(
    file: UploadFile = File(...),
    color_name: str = Form(..., description="Hair color name (e.g., Blonde, Brown, etc.)"),
    response_format: str = Form("json", description="Response format: 'json' for base64 images or 'zip' for downloadable ZIP"),
    color_change_service: ColorChangeService = Depends(get_color_change_service)
):
    """
    Change hair color with base color + all available tones
    
    Args:
        file: Image file (jpg, png, etc.)
        color_name: Hair color name from available colors
        response_format: 'json' for base64 images or 'zip' for downloadable ZIP file
        
    Returns:
        JSON response with base64 images OR ZIP file download (based on response_format)
    """
    try:
        # Use color change service
        result_bytes_dict = color_change_service.change_hair_color_with_all_tones_file(file, color_name)
        
        if response_format.lower() == "zip":
            # Create ZIP file in memory
            zip_buffer = io.BytesIO()
            
            with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                # Add base result
                base_filename = f"{color_name.lower()}_base_{file.filename.split('.')[0]}.png"
                zip_file.writestr(base_filename, result_bytes_dict['base_result'])
                
                # Add all tones
                for tone_name, tone_bytes in result_bytes_dict['tones'].items():
                    if tone_bytes is not None:
                        tone_filename = f"{color_name.lower()}_{tone_name.lower()}_{file.filename.split('.')[0]}.png"
                        zip_file.writestr(tone_filename, tone_bytes)
            
            zip_buffer.seek(0)
            zip_filename = f"hair_color_variations_{color_name.lower()}_{file.filename.split('.')[0]}.zip"
            
            # Return ZIP as response
            return Response(
                content=zip_buffer.getvalue(),
                media_type="application/zip",
                headers={"Content-Disposition": f"attachment; filename={zip_filename}"}
            )
        else:
            # Return JSON with base64 images (default behavior)
            import base64
            
            response_data = {
                "success": True,
                "color": color_name,
                "response_format": "json",
                "download_links": {
                    "base": f"/change-hair-color-base-only/{color_name}",
                    "tones": {}
                },
                "base_result": base64.b64encode(result_bytes_dict['base_result']).decode('utf-8'),
                "tones": {}
            }
            
            for tone_name, tone_bytes in result_bytes_dict['tones'].items():
                if tone_bytes is not None:
                    response_data['tones'][tone_name] = base64.b64encode(tone_bytes).decode('utf-8')
                    response_data['download_links']['tones'][tone_name] = f"/change-hair-color-tone-only/{color_name}/{tone_name}"
                else:
                    response_data['tones'][tone_name] = None
                    response_data['download_links']['tones'][tone_name] = None
            
            return response_data
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Color change with all tones failed: {str(e)}"
        )


@router.post("/upload-and-prepare")
async def upload_and_prepare(
    file: UploadFile = File(...),
    color_change_service: ColorChangeService = Depends(get_color_change_service)
):
    """
    Upload image, validate it and prepare for fast color changes
    
    Args:
        file: Image file (jpg, png, etc.)
        
    Returns:
        Session ID for subsequent fast operations
    """
    try:
        session_id = color_change_service.upload_and_prepare_image(file)
        
        return {
            "success": True,
            "session_id": session_id,
            "message": "Image uploaded and prepared successfully",
            "expires_in_minutes": 30
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Image upload failed: {str(e)}"
        )


@router.post("/change-hair-color-fast/{session_id}")
async def change_hair_color_fast(
    session_id: str,
    color_name: str = Form(..., description="Hair color name (e.g., Blonde, Brown, etc.)"),
    tone: Optional[str] = Form(None, description="Optional tone for the color (e.g., golden, ash, etc.)"),
    color_change_service: ColorChangeService = Depends(get_color_change_service)
):
    """
    Fast hair color change using cached session data (NO mask generation!)
    
    Args:
        session_id: Session identifier from upload-and-prepare
        color_name: Hair color name from available colors
        tone: Optional tone name for the color
        
    Returns:
        Color-changed image file for download
    """
    try:
        # Use fast color change service
        result_bytes = color_change_service.change_hair_color_with_session(session_id, color_name, tone)
        
        # Create filename
        tone_suffix = f"_{tone}" if tone else ""
        filename = f"fast_color_{color_name.lower()}{tone_suffix}_{session_id}.png"
        
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
            detail=f"Fast color change failed: {str(e)}"
        )


@router.post("/change-hair-color-all-tones-fast/{session_id}")
async def change_hair_color_all_tones_fast(
    session_id: str,
    color_name: str = Form(..., description="Hair color name (e.g., Blonde, Brown, etc.)"),
    color_change_service: ColorChangeService = Depends(get_color_change_service)
):
    """
    Fast hair color change with ALL tones using cached session data (NO mask generation!)
    
    Args:
        session_id: Session identifier from upload-and-prepare
        color_name: Hair color name from available colors
        
    Returns:
        JSON response with base64 images for base + all tones
    """
    try:
        # Use fast color change service for all tones
        import base64
        
        # Get base color result
        base_result_bytes = color_change_service.change_hair_color_with_session(session_id, color_name, None)
        
        # Get all available tones for this color
        available_tones = color_change_service.get_available_tones(color_name)
        
        response_data = {
            "success": True,
            "color": color_name,
            "session_id": session_id,
            "base_result": base64.b64encode(base_result_bytes).decode('utf-8'),
            "tones": {}
        }
        
        # Get result for each tone
        for tone in available_tones:
            try:
                tone_result_bytes = color_change_service.change_hair_color_with_session(session_id, color_name, tone)
                response_data['tones'][tone] = base64.b64encode(tone_result_bytes).decode('utf-8')
            except Exception as e:
                print(f"Failed to process tone {tone}: {e}")
                response_data['tones'][tone] = None
        
        return response_data
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Fast color change with all tones failed: {str(e)}"
        )


@router.delete("/cleanup-session/{session_id}")
async def cleanup_session(
    session_id: str,
    color_change_service: ColorChangeService = Depends(get_color_change_service)
):
    """
    Clean up session data
    
    Args:
        session_id: Session identifier to cleanup
        
    Returns:
        Success message
    """
    try:
        color_change_service._cleanup_session(session_id)
        
        return {
            "success": True,
            "message": f"Session {session_id} cleaned up successfully"
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Session cleanup failed: {str(e)}"
        )