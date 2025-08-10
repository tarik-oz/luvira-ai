"""
Prediction service for hair segmentation
"""

import tempfile
import logging
import os
import cv2
import numpy as np

from ..core.exceptions import PredictionException, ImageProcessingException
from ..utils import FileValidator, ImageUtils

logger = logging.getLogger(__name__)


class PredictionService:
    """Service for handling prediction operations"""
    
    def __init__(self, model_service):
        self.model_service = model_service
    
    def predict_mask_file(self, file) -> bytes:
        """
        Predict hair mask and return as file bytes
        
        Args:
            file: Uploaded file
            
        Returns:
            Mask image bytes
        """
        # Validate file
        FileValidator.validate_upload_file(file)
        
        # Check if model is loaded
        if not self.model_service.is_model_loaded():
            raise PredictionException("Model is not loaded")
        
        temp_input_path = None
        
        try:
            # Read and process image
            image = ImageUtils.process_uploaded_file(file)
            
            # Create temporary file for prediction (use PNG to avoid JPEG artifacts on masks)
            temp_input_path = self._create_temp_image_file(image)
            
            # Make prediction
            predictor = self.model_service.get_predictor()
            if predictor is None:
                raise PredictionException("Predictor not available")
            
            # Get prediction directly
            _, predicted_mask, _ = predictor.predict(temp_input_path)
            
            if predicted_mask is None:
                raise PredictionException("Failed to generate mask")
            
            # Convert mask to bytes
            mask_bytes = ImageUtils.mask_to_bytes(predicted_mask)
            
            logger.info(f"Successfully generated mask for file: {file.filename}")
            return mask_bytes
            
        except Exception as e:
            logger.error(f"Prediction failed for file {file.filename}: {str(e)}")
            raise PredictionException(f"Prediction failed: {str(e)}")
        finally:
            # Cleanup
            ImageUtils.cleanup_temp_file(temp_input_path)
    
    def _create_temp_image_file(self, image: np.ndarray) -> str:
        """Create temporary image file"""
        try:
            # Create temp file with proper suffix
            temp_fd, temp_input_path = tempfile.mkstemp(suffix='.png')
            
            try:
                # Close the file descriptor since we'll use cv2.imwrite
                os.close(temp_fd)
                
                # Save temporary image (PNG, lossless)
                cv2.imwrite(temp_input_path, image)
                return temp_input_path
                
            except Exception as e:
                # Cleanup temp file if cv2.imwrite fails
                try:
                    os.unlink(temp_input_path)
                except:
                    pass
                raise e
            
        except Exception as e:
            logger.error(f"Failed to create temp file: {str(e)}")
            raise ImageProcessingException(f"Failed to create temporary file: {str(e)}") 