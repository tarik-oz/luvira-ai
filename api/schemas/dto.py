"""
Data Transfer Objects (DTOs) for Hair Segmentation API

This module contains Pydantic models for request/response validation
and API documentation.
"""

from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field
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


class ColorChangeRequest(BaseModel):
    """Color change request DTO (legacy RGB)"""
    target_color: List[int] = Field(..., description="Target hair color in RGB format [R, G, B]", min_items=3, max_items=3)
    
    model_config = {"protected_namespaces": ()}


class ColorChangeByNameRequest(BaseModel):
    """Color change by name request DTO"""
    color_name: str = Field(..., description="Hair color name from available colors")
    tone: Optional[str] = Field(None, description="Optional tone for the color")
    
    model_config = {"protected_namespaces": ()}


class AvailableColorsResponse(BaseModel):
    """Available colors response DTO"""
    success: bool = Field(default=True, description="Operation success status")
    colors: List[str] = Field(..., description="List of available color names")
    count: int = Field(..., description="Number of available colors")


class AvailableTonesResponse(BaseModel):
    """Available tones response DTO"""
    success: bool = Field(default=True, description="Operation success status")
    color: str = Field(..., description="Color name")
    tones: List[str] = Field(..., description="List of available tone names for the color")
    count: int = Field(..., description="Number of available tones")


class ColorChangeAllTonesResponse(BaseModel):
    """Color change all tones response DTO"""
    success: bool = Field(default=True, description="Operation success status")
    color: str = Field(..., description="Base color name")
    base_result: str = Field(..., description="Base color result as base64 encoded image")
    tones: Dict[str, Optional[str]] = Field(..., description="Dictionary of tone results as base64 encoded images")
