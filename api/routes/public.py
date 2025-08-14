"""
Frontend-specific routes for Hair Segmentation API
These endpoints are optimized for frontend applications
"""

from fastapi import APIRouter, File, UploadFile, HTTPException, Depends, Form, Request
from fastapi.responses import Response, StreamingResponse
from ..core import get_color_change_service, SessionExpiredException
from ..core.exceptions import APIException
from ..services import ColorChangeService

import json

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
        
    except APIException as e:
        # Return structured error with code for frontend mapping
        payload = {"detail": e.detail, "error_code": e.error_code, "extra": getattr(e, "extra_data", {})}
        return Response(content=json.dumps(payload), media_type="application/json", status_code=e.status_code)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Image upload failed: {str(e)}")

@router.post("/overlays-with-session/{session_id}")
async def overlays_with_session(
    request: Request,
    session_id: str,
    color_name: str = Form(..., description="Hair color name (e.g., Blonde, Brown, etc.)"),
    color_change_service: ColorChangeService = Depends(get_color_change_service)
):
    """
    Stream WEBP overlays (multipart/mixed) for base + all tones using cached session data.
    Part names: base, then each tone name. Quality: WEBP q=80. Soft edges + alpha.
    """
    try:
        boundary = "luvira"

        def iter_multipart():
            try:
                for name, webp_bytes in color_change_service.iter_overlays_with_session(
                    session_id, color_name, webp_quality=80
                ):
                    yield f"--{boundary}\r\n".encode()
                    yield b"Content-Type: image/webp\r\n"
                    yield f"Content-Disposition: attachment; name=\"{name}\"; filename=\"{name}.webp\"\r\n\r\n".encode()
                    yield webp_bytes + b"\r\n"
                yield f"--{boundary}--\r\n".encode()
            except SessionExpiredException as e:
                # Encode JSON error as single-part JSON for consistency
                import json
                payload = {
                    "detail": e.detail,
                    "error_code": e.error_code,
                    "extra": getattr(e, "extra_data", {}),
                }
                yield f"--{boundary}\r\n".encode()
                yield b"Content-Type: application/json\r\n\r\n"
                yield json.dumps(payload).encode()
                yield b"\r\n"
                yield f"--{boundary}--\r\n".encode()

        return StreamingResponse(
            iter_multipart(),
            media_type=f"multipart/mixed; boundary={boundary}",
            headers={
                # Hint clients not to buffer the whole response
                "X-Accel-Buffering": "no",
                "Cache-Control": "no-cache",
            },
        )
    except SessionExpiredException as e:
        import json
        payload = {
            "detail": e.detail,
            "error_code": e.error_code,
            "extra": getattr(e, "extra_data", {}),
        }
        return Response(
            content=json.dumps(payload),
            media_type="application/json",
            status_code=e.status_code,
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Overlay stream failed: {str(e)}")