"""
Custom exceptions for Hair Segmentation API
"""

from typing import Optional, Dict, Any
from fastapi import HTTPException


class APIException(HTTPException):
    """Base API exception class"""
    
    def __init__(
        self,
        status_code: int,
        detail: str,
        error_code: Optional[str] = None,
        extra_data: Optional[Dict[str, Any]] = None
    ):
        super().__init__(status_code=status_code, detail=detail)
        self.error_code = error_code
        self.extra_data = extra_data or {}


class ModelNotLoadedException(APIException):
    """Raised when model is not loaded"""
    
    def __init__(self, detail: str = "Model is not loaded"):
        super().__init__(
            status_code=500,
            detail=detail,
            error_code="MODEL_NOT_LOADED"
        )


class ModelLoadException(APIException):
    """Raised when model fails to load"""
    
    def __init__(self, model_path: str, detail: str = "Failed to load model"):
        super().__init__(
            status_code=500,
            detail=f"{detail}: {model_path}",
            error_code="MODEL_LOAD_FAILED",
            extra_data={"model_path": model_path}
        )


class FileValidationException(APIException):
    """Raised when file validation fails"""
    
    def __init__(self, detail: str, error_code: str = "FILE_VALIDATION_FAILED"):
        super().__init__(
            status_code=400,
            detail=detail,
            error_code=error_code
        )


class PredictionException(APIException):
    """Raised when prediction fails"""
    
    def __init__(self, detail: str = "Prediction failed"):
        super().__init__(
            status_code=500,
            detail=detail,
            error_code="PREDICTION_FAILED"
        )


class ImageProcessingException(APIException):
    """Raised when image processing fails"""
    
    def __init__(self, detail: str = "Image processing failed"):
        super().__init__(
            status_code=400,
            detail=detail,
            error_code="IMAGE_PROCESSING_FAILED"
        )


# Error response helper
def create_error_response(
    status_code: int,
    detail: str,
    error_code: Optional[str] = None,
    extra_data: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Create standardized error response"""
    response = {
        "success": False,
        "error": detail,
        "status_code": status_code
    }
    
    if error_code:
        response["error_code"] = error_code
    
    if extra_data:
        response["extra_data"] = extra_data
    
    return response 