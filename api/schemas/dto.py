"""
Data Transfer Objects (DTOs) for Hair Segmentation API

This module contains Pydantic models for request/response validation
and API documentation.
"""

from typing import Optional, Dict, List
from pydantic import BaseModel, Field
from enum import Enum


class HealthStatus(str, Enum):
    """Health status enumeration"""
    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"
    DEGRADED = "degraded"


class HealthCheckResponse(BaseModel):
    """Health check response DTO"""
    status: HealthStatus = Field(..., description="System health status")
    model_loaded: bool = Field(..., description="Whether the model is loaded")
    model_path: Optional[str] = Field(None, description="Path to the loaded model")
    version: Optional[str] = Field(None, description="API version")


class ModelInfoResponse(BaseModel):
    """Model information response DTO"""
    model_path: Optional[str] = Field(None, description="Path to the model file")
    model_type: Optional[str] = Field(None, description="Type of the model")
    input_shape: Optional[List[int]] = Field(None, description="Model input shape")
    output_shape: Optional[List[int]] = Field(None, description="Model output shape")
    device: Optional[str] = Field(None, description="Device the model is running on")
    is_loaded: bool = Field(..., description="Whether the model is currently loaded")


class ReloadModelRequest(BaseModel):
    """Model reload request DTO"""
    model_path: Optional[str] = Field(None, description="Path to the model file")


class ReloadModelResponse(BaseModel):
    """Model reload response DTO"""
    success: bool = Field(..., description="Operation success status")
    message: str = Field(..., description="Operation message")
    model_info: ModelInfoResponse = Field(..., description="Updated model information")


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


class ColorChangeRequest(BaseModel):
    """Color change request DTO (legacy RGB)"""
    target_color: List[int] = Field(..., description="Target hair color in RGB format [R, G, B]", min_items=3, max_items=3)
