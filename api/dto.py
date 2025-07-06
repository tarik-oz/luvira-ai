"""
Data Transfer Objects (DTOs) for Hair Segmentation API

This module contains Pydantic models for request/response validation
and API documentation.
"""

from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field, validator
from enum import Enum


class HealthStatus(str, Enum):
    """Health status enumeration"""
    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"
    DEGRADED = "degraded"


class ErrorResponse(BaseModel):
    """Standard error response DTO"""
    success: bool = Field(default=False, description="Operation success status")
    error: str = Field(..., description="Error message")
    detail: Optional[str] = Field(None, description="Detailed error information")
    status_code: int = Field(..., description="HTTP status code")


class SuccessResponse(BaseModel):
    """Standard success response DTO"""
    success: bool = Field(default=True, description="Operation success status")
    message: str = Field(..., description="Success message")


class HealthCheckResponse(BaseModel):
    """Health check response DTO"""
    status: HealthStatus = Field(..., description="System health status")
    model_loaded: bool = Field(..., description="Whether the model is loaded")
    model_path: Optional[str] = Field(None, description="Path to the loaded model")
    version: Optional[str] = Field(None, description="API version")
    
    model_config = {"protected_namespaces": ()}


class ModelInfoResponse(BaseModel):
    """Model information response DTO"""
    model_path: Optional[str] = Field(None, description="Path to the model file")
    model_type: Optional[str] = Field(None, description="Type of the model")
    input_shape: Optional[List[int]] = Field(None, description="Model input shape")
    output_shape: Optional[List[int]] = Field(None, description="Model output shape")
    device: Optional[str] = Field(None, description="Device the model is running on")
    is_loaded: bool = Field(..., description="Whether the model is currently loaded")
    
    model_config = {"protected_namespaces": ()}


class ReloadModelRequest(BaseModel):
    """Model reload request DTO"""
    model_path: Optional[str] = Field(None, description="Path to the model file")
    
    model_config = {"protected_namespaces": ()}


class ReloadModelResponse(BaseModel):
    """Model reload response DTO"""
    success: bool = Field(..., description="Operation success status")
    message: str = Field(..., description="Operation message")
    model_info: ModelInfoResponse = Field(..., description="Updated model information")
    
    model_config = {"protected_namespaces": ()}


class PredictMaskResponse(BaseModel):
    """Mask prediction response DTO"""
    success: bool = Field(..., description="Operation success status")
    mask_base64: str = Field(..., description="Base64 encoded mask image")
    original_filename: str = Field(..., description="Original uploaded filename")
    mask_shape: List[int] = Field(..., description="Shape of the predicted mask")
    confidence_score: float = Field(..., description="Prediction confidence score")
    
    @validator('confidence_score')
    def validate_confidence_score(cls, v):
        """Validate confidence score is between 0 and 1"""
        if not 0 <= v <= 1:
            raise ValueError('Confidence score must be between 0 and 1')
        return v


class ClearModelResponse(BaseModel):
    """Clear model response DTO"""
    success: bool = Field(..., description="Operation success status")
    message: str = Field(..., description="Operation message")


class RootResponse(BaseModel):
    """Root endpoint response DTO"""
    message: str = Field(..., description="API welcome message")
    version: str = Field(..., description="API version")
    model_loaded: bool = Field(..., description="Whether the model is loaded")
    endpoints: Dict[str, str] = Field(..., description="Available API endpoints")
    
    model_config = {"protected_namespaces": ()}


# File upload validation
class FileUploadValidator:
    """File upload validation helper"""
    
    ALLOWED_IMAGE_TYPES = ["image/jpeg", "image/png", "image/jpg"]
    MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
    
    @classmethod
    def validate_file_type(cls, content_type: str) -> bool:
        """Validate file content type"""
        return content_type in cls.ALLOWED_IMAGE_TYPES
    
    @classmethod
    def validate_file_size(cls, file_size: Optional[int]) -> bool:
        """Validate file size"""
        if file_size is None:
            return True  # Allow if size is unknown
        return file_size <= cls.MAX_FILE_SIZE
    
    @classmethod
    def get_validation_error(cls, content_type: str, file_size: Optional[int]) -> Optional[str]:
        """Get validation error message if any"""
        if not cls.validate_file_type(content_type):
            return f"File type not allowed. Allowed types: {cls.ALLOWED_IMAGE_TYPES}"
        
        if not cls.validate_file_size(file_size):
            return f"File too large. Maximum size: {cls.MAX_FILE_SIZE // (1024*1024)}MB"
        
        return None 