"""
Prediction service for hair segmentation
"""

import tempfile
import logging
import threading
import time
from pathlib import Path
from typing import Tuple, Optional
import cv2
import numpy as np

from ..core.exceptions import PredictionException, ImageProcessingException
from ..utils import FileValidator
from ..config import MODEL_CONFIG

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
            image = self._process_uploaded_image(file)
            
            # Create temporary file for prediction
            temp_input_path = self._create_temp_image_file(image)
            
            # Make prediction with timeout
            predictor = self.model_service.get_predictor()
            timeout = MODEL_CONFIG.get("prediction_timeout", 30)
            
            # Run prediction with timeout
            predicted_mask = self._make_prediction_with_timeout(temp_input_path, predictor, timeout)
            
            # Convert mask to bytes
            mask_bytes = self._mask_to_bytes(predicted_mask)
            
            logger.info(f"Successfully generated mask for file: {file.filename}")
            return mask_bytes
            
        except Exception as e:
            logger.error(f"Prediction failed for file {file.filename}: {str(e)}")
            raise PredictionException(f"Prediction failed: {str(e)}")
        finally:
            # Cleanup
            self._cleanup_temp_file(temp_input_path)
    
    def _process_uploaded_image(self, file) -> np.ndarray:
        """Process uploaded image file"""
        try:
            # Read image file
            image_data = file.file.read()
            
            # Convert to numpy array
            nparr = np.frombuffer(image_data, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if image is None:
                raise ImageProcessingException("Could not decode image")
            
            # Validate image dimensions and format
            FileValidator.validate_image_dimensions(image)
            FileValidator.validate_image_format(image)
            
            return image
            
        except Exception as e:
            logger.error(f"Image processing failed: {str(e)}")
            raise ImageProcessingException(f"Image processing failed: {str(e)}")
    
    def _create_temp_image_file(self, image: np.ndarray) -> str:
        """Create temporary image file"""
        try:
            with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as temp_input:
                temp_input_path = temp_input.name
            
            # Save temporary image
            cv2.imwrite(temp_input_path, image)
            return temp_input_path
            
        except Exception as e:
            logger.error(f"Failed to create temp file: {str(e)}")
            raise ImageProcessingException(f"Failed to create temporary file: {str(e)}")
    
    def _make_prediction_with_timeout(self, temp_input_path: str, predictor, timeout: int) -> np.ndarray:
        """Make prediction using model with timeout (Windows compatible)"""
        try:
            if predictor is None:
                raise PredictionException("Predictor not available")
            
            # Thread-safe result storage
            result = {"mask": None, "error": None, "completed": False}
            
            def prediction_worker():
                """Worker function to run prediction in separate thread"""
                try:
                    original_image, predicted_mask, binary_mask = predictor.predict(temp_input_path)
                    
                    if predicted_mask is None:
                        result["error"] = "Failed to generate mask"
                    else:
                        result["mask"] = predicted_mask
                        
                except Exception as e:
                    result["error"] = str(e)
                finally:
                    result["completed"] = True
            
            # Start prediction in separate thread
            prediction_thread = threading.Thread(target=prediction_worker)
            prediction_thread.daemon = True
            prediction_thread.start()
            
            # Wait for completion or timeout
            start_time = time.time()
            while not result["completed"]:
                if time.time() - start_time > timeout:
                    raise TimeoutError(f"Prediction timed out after {timeout} seconds")
                time.sleep(0.1)  # Small delay to avoid busy waiting
            
            # Check for errors
            if result["error"]:
                raise Exception(result["error"])
            
            return result["mask"]
                
        except TimeoutError:
            raise PredictionException(f"Prediction timed out after {timeout} seconds")
        except Exception as e:
            logger.error(f"Model prediction failed: {str(e)}")
            raise PredictionException(f"Model prediction failed: {str(e)}")
    
    def _mask_to_bytes(self, predicted_mask: np.ndarray) -> bytes:
        """Convert mask to bytes"""
        try:
            mask_255 = (predicted_mask * 255).astype(np.uint8)
            _, buffer = cv2.imencode('.png', mask_255)
            return buffer.tobytes()
        except Exception as e:
            logger.error(f"Failed to convert mask to bytes: {str(e)}")
            raise ImageProcessingException(f"Failed to convert mask to bytes: {str(e)}")
    
    def _cleanup_temp_file(self, temp_path: Optional[str]) -> None:
        """Clean up temporary file"""
        if temp_path:
            try:
                Path(temp_path).unlink(missing_ok=True)
            except Exception as e:
                logger.warning(f"Failed to cleanup temp file {temp_path}: {str(e)}") 