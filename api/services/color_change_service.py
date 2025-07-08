"""
Color change service for hair color modification
"""

import tempfile
import logging
import threading
import time
from pathlib import Path
from typing import Tuple, Optional
import cv2
import numpy as np
import sys
import os

# Add color_changer module to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'color_changer'))
from color_changer import HairColorChanger

from ..core.exceptions import PredictionException, ImageProcessingException
from ..utils import FileValidator, ColorValidator
from ..config import MODEL_CONFIG

logger = logging.getLogger(__name__)


class ColorChangeService:
    """Service for handling hair color change operations"""
    
    def __init__(self, prediction_service):
        self.prediction_service = prediction_service
        self.color_changer = HairColorChanger()
    
    def change_hair_color_file(self, file, target_color: list) -> bytes:
        """
        Change hair color and return as file bytes
        
        Args:
            file: Uploaded image file
            target_color: Target hair color [R, G, B] (0-255)
            
        Returns:
            Color-changed image bytes
        """
        # Validate file
        FileValidator.validate_upload_file(file)
        
        # Validate target color using ColorValidator
        ColorValidator.validate_rgb_color(target_color)
        
        # Check if model is loaded for mask prediction
        if not self.prediction_service.model_service.is_model_loaded():
            raise PredictionException("Model is not loaded - cannot generate hair mask")
        
        temp_input_path = None
        
        try:
            # Read image data once and store it
            image_data = file.file.read()
            file.file.seek(0)  # Reset file pointer to beginning
            
            # Process image from data
            original_image = self._process_image_data(image_data)
            
            # Generate hair mask using prediction service with timeout handling
            try:
                mask_bytes = self.prediction_service.predict_mask_file(file)
            except Exception as e:
                if "timed out" in str(e).lower():
                    raise ImageProcessingException("Mask generation timed out. Please try with a smaller image or try again later.")
                else:
                    raise ImageProcessingException(f"Mask generation failed: {str(e)}")
            
            # Convert mask bytes to numpy array
            mask_array = self._bytes_to_mask(mask_bytes)
            
            # Apply color change with timeout
            result_image = self._apply_color_change_with_timeout(original_image, mask_array, target_color)
            
            # Convert result to bytes
            result_bytes = self._image_to_bytes(result_image)
            
            logger.info(f"Successfully changed hair color for file: {file.filename}")
            return result_bytes
            
        except Exception as e:
            logger.error(f"Color change failed for file {file.filename}: {str(e)}")
            raise ImageProcessingException(f"Color change failed: {str(e)}")
        finally:
            # Cleanup
            self._cleanup_temp_file(temp_input_path)
    
    def _process_image_data(self, image_data: bytes) -> np.ndarray:
        """Process image data from bytes"""
        try:
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
    
    def _bytes_to_mask(self, mask_bytes: bytes) -> np.ndarray:
        """Convert mask bytes to numpy array"""
        try:
            nparr = np.frombuffer(mask_bytes, np.uint8)
            mask = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
            
            if mask is None:
                raise ImageProcessingException("Could not decode mask")
            
            return mask
            
        except Exception as e:
            logger.error(f"Failed to convert mask bytes to array: {str(e)}")
            raise ImageProcessingException(f"Failed to convert mask bytes to array: {str(e)}")
    
    def _image_to_bytes(self, image: np.ndarray) -> bytes:
        """Convert image to bytes"""
        try:
            # Convert RGB to BGR for OpenCV
            image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            _, buffer = cv2.imencode('.png', image_bgr)
            return buffer.tobytes()
        except Exception as e:
            logger.error(f"Failed to convert image to bytes: {str(e)}")
            raise ImageProcessingException(f"Failed to convert image to bytes: {str(e)}")
    
    def _cleanup_temp_file(self, temp_path: Optional[str]) -> None:
        """Clean up temporary file"""
        if temp_path:
            try:
                Path(temp_path).unlink(missing_ok=True)
            except Exception as e:
                logger.warning(f"Failed to cleanup temp file {temp_path}: {str(e)}") 
    
    def _apply_color_change_with_timeout(self, original_image: np.ndarray, mask_array: np.ndarray, target_color: list) -> np.ndarray:
        """Apply color change with timeout"""
        try:
            # Thread-safe result storage
            result = {"image": None, "error": None, "completed": False}
            
            def color_change_worker():
                """Worker function to run color change in separate thread"""
                try:
                    result_image = self.color_changer.change_hair_color(original_image, mask_array, target_color)
                    result["image"] = result_image
                except Exception as e:
                    result["error"] = str(e)
                finally:
                    result["completed"] = True
            
            # Start color change in separate thread
            color_change_thread = threading.Thread(target=color_change_worker)
            color_change_thread.daemon = True
            color_change_thread.start()
            
            # Wait for completion or timeout
            timeout = MODEL_CONFIG.get("color_change_timeout", 10)  # 10 seconds default
            start_time = time.time()
            while not result["completed"]:
                if time.time() - start_time > timeout:
                    raise TimeoutError(f"Color change timed out after {timeout} seconds")
                time.sleep(0.1)  # Small delay to avoid busy waiting
            
            # Check for errors
            if result["error"]:
                raise Exception(result["error"])
            
            return result["image"]
                
        except TimeoutError:
            raise ImageProcessingException(f"Color change timed out after {timeout} seconds. Please try with a smaller image or try again later.")
        except Exception as e:
            logger.error(f"Color change processing failed: {str(e)}")
            raise ImageProcessingException(f"Color change processing failed: {str(e)}") 