"""
Model management routes for Hair Segmentation API
"""

from fastapi import APIRouter, HTTPException, Depends
from ..schemas import ModelInfoResponse, ReloadModelRequest, ReloadModelResponse, ClearModelResponse
from ..core import get_model_service
from ..services import ModelService

router = APIRouter()


@router.get("/model-info", response_model=ModelInfoResponse)
async def get_model_info(model_service: ModelService = Depends(get_model_service)):
    """Get information about the loaded model"""
    model_info = model_service.get_model_info()
    
    return ModelInfoResponse(
        model_path=model_info.get("model_path"),
        model_type=model_info.get("model_type"),
        input_shape=model_info.get("input_shape"),
        output_shape=model_info.get("output_shape"),
        device=model_info.get("device"),
        is_loaded=model_service.is_model_loaded()
    )


@router.post("/reload-model", response_model=ReloadModelResponse)
async def reload_model(request: ReloadModelRequest, model_service: ModelService = Depends(get_model_service)):
    """
    Reload the model from a different path
    
    Args:
        request: ReloadModelRequest containing model path
        
    Returns:
        ReloadModelResponse with success status and model info
    """
    try:
        success = model_service.reload_model(request.model_path)
        if success:
            model_info = model_service.get_model_info()
            return ReloadModelResponse(
                success=True,
                message=f"Model reloaded successfully",
                model_info=ModelInfoResponse(
                    model_path=model_info.get("model_path"),
                    model_type=model_info.get("model_type"),
                    input_shape=model_info.get("input_shape"),
                    output_shape=model_info.get("output_shape"),
                    device=model_info.get("device"),
                    is_loaded=model_service.is_model_loaded()
                )
            )
        else:
            raise HTTPException(
                status_code=500,
                detail="Failed to reload model"
            )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error reloading model: {str(e)}"
        )


@router.post("/clear-model", response_model=ClearModelResponse)
async def clear_model(model_service: ModelService = Depends(get_model_service)):
    """Clear the loaded model from memory"""
    try:
        model_service.clear_model()
        return ClearModelResponse(
            success=True,
            message="Model cleared from memory"
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error clearing model: {str(e)}"
        ) 