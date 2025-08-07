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


class SessionExpiredException(APIException):
    """Raised when session has expired or doesn't exist"""
    
    def __init__(self, session_id: str, detail: str = "Session has expired or doesn't exist"):
        super().__init__(
            status_code=404,
            detail=f"{detail}: {session_id}",
            error_code="SESSION_EXPIRED",
            extra_data={"session_id": session_id}
        ) 