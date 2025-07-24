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
from color_changer import ColorTransformer

from ..core.exceptions import PredictionException, ImageProcessingException
from ..utils import FileValidator, ColorValidator
from ..config import MODEL_CONFIG

logger = logging.getLogger(__name__)


class ColorChangeService:
    """Service for handling hair color change operations"""
    
    def __init__(self, prediction_service):
        self.prediction_service = prediction_service
        self.color_transformer = ColorTransformer()
    
    @staticmethod
    def get_available_colors() -> list:
        """
        Get list of available colors from config
        
        Returns:
            List of available color names
        """
        # Import here to avoid circular import
        from color_changer.config.color_config import COLORS
        return [name for _, name in COLORS]
    
    @staticmethod
    def get_available_tones(color_name: str) -> list:
        """
        Get list of available tones for a specific color
        
        Args:
            color_name: Name of the color
            
        Returns:
            List of available tone names for the color
        """
        # Import here to avoid circular import
        from color_changer.config.color_config import CUSTOM_TONES
        
        # Get correct case color name
        from ..utils.validators import ColorValidator
        correct_color_name = ColorValidator.get_correct_color_name(color_name)
        
        return list(CUSTOM_TONES.get(correct_color_name, {}).keys())
    
    def change_hair_color_by_name(self, file, color_name: str, tone: str = None) -> bytes:
        """
        Change hair color using color name and optional tone
        
        Args:
            file: Uploaded image file
            color_name: Color name from COLORS config (e.g., "Blonde", "Brown")
            tone: Optional tone name (e.g., "golden", "ash")
            
        Returns:
            Color-changed image bytes
        """
        # Validate file
        FileValidator.validate_upload_file(file)
        
        # Validate color name
        ColorValidator.validate_color_name(color_name)
        correct_color_name = ColorValidator.get_correct_color_name(color_name)
        
        # Validate tone if provided
        if tone:
            ColorValidator.validate_tone_name(color_name, tone)
            correct_tone = ColorValidator.get_correct_tone_name(color_name, tone)
        else:
            correct_tone = None
        
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
            if correct_tone:
                result_image = self._apply_color_change_with_tone_with_timeout(
                    original_image, mask_array, correct_color_name, correct_tone
                )
            else:
                result_image = self._apply_color_change_by_name_with_timeout(
                    original_image, mask_array, correct_color_name
                )
            
            # Convert result to bytes
            result_bytes = self._image_to_bytes(result_image)
            
            logger.info(f"Successfully changed hair color to {correct_color_name}{f' ({correct_tone})' if correct_tone else ''} for file: {file.filename}")
            return result_bytes
            
        except Exception as e:
            logger.error(f"Color change failed for file {file.filename}: {str(e)}")
            raise ImageProcessingException(f"Color change failed: {str(e)}")
        finally:
            # Cleanup
            self._cleanup_temp_file(temp_input_path)
    
    def change_hair_color_file(self, file, target_color: list) -> bytes:
        """
        Change hair color and return as file bytes (legacy RGB method)
        
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
    
    def change_hair_color_with_all_tones_file(self, file, color_name: str) -> dict:
        """
        Change hair color with base color + all tones and return as dict
        
        Args:
            file: Uploaded image file
            color_name: Color name from COLORS config (e.g., "Blonde", "Brown")
            
        Returns:
            Dictionary with base result and all tones as bytes:
            {
                'base_result': bytes,
                'tones': {
                    'golden': bytes,
                    'ash': bytes,
                    ...
                }
            }
        """
        # Validate file
        FileValidator.validate_upload_file(file)
        
        # Validate color name
        ColorValidator.validate_color_name(color_name)
        correct_color_name = ColorValidator.get_correct_color_name(color_name)
        
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
            
            # Apply color change with all tones (with timeout)
            result_images = self._apply_color_change_with_all_tones_with_timeout(
                original_image, mask_array, correct_color_name
            )
            
            # Convert all results to bytes
            result_bytes = {}
            result_bytes['base_result'] = self._image_to_bytes(result_images['base_result'])
            result_bytes['tones'] = {}
            
            for tone_name, tone_image in result_images['tones'].items():
                if tone_image is not None:
                    result_bytes['tones'][tone_name] = self._image_to_bytes(tone_image)
                else:
                    result_bytes['tones'][tone_name] = None
            
            logger.info(f"Successfully changed hair color with all tones for file: {file.filename}")
            return result_bytes
            
        except Exception as e:
            logger.error(f"Color change with tones failed for file {file.filename}: {str(e)}")
            raise ImageProcessingException(f"Color change with tones failed: {str(e)}")
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
                    result_image = self.color_transformer.change_hair_color(original_image, mask_array, target_color)
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
    
    def _apply_color_change_by_name_with_timeout(self, original_image: np.ndarray, mask_array: np.ndarray, color_name: str) -> np.ndarray:
        """Apply color change by name with timeout"""
        try:
            # Thread-safe result storage
            result = {"image": None, "error": None, "completed": False}
            
            def color_change_worker():
                """Worker function to run color change in separate thread"""
                try:
                    result_image = self.color_transformer.change_hair_color(original_image, mask_array, color_name)
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
    
    def _apply_color_change_with_tone_with_timeout(self, original_image: np.ndarray, mask_array: np.ndarray, color_name: str, tone: str) -> np.ndarray:
        """Apply color change with specific tone with timeout"""
        try:
            # Thread-safe result storage
            result = {"image": None, "error": None, "completed": False}
            
            def color_change_worker():
                """Worker function to run color change in separate thread"""
                try:
                    result_image = self.color_transformer.apply_color_with_tone(original_image, mask_array, color_name, tone)
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
    
    def _apply_color_change_with_all_tones_with_timeout(
        self, 
        original_image: np.ndarray, 
        mask_array: np.ndarray, 
        color_name: str
    ) -> dict:
        """Apply color change with all tones with timeout"""
        try:
            # Thread-safe result storage
            result = {"images": None, "error": None, "completed": False}
            
            def color_change_worker():
                """Worker function to run color change with all tones in separate thread"""
                try:
                    result_images = self.color_transformer.change_hair_color_with_all_tones(
                        original_image, mask_array, color_name
                    )
                    result["images"] = result_images
                except Exception as e:
                    result["error"] = str(e)
                finally:
                    result["completed"] = True
            
            # Start color change in separate thread
            color_change_thread = threading.Thread(target=color_change_worker)
            color_change_thread.daemon = True
            color_change_thread.start()
            
            # Wait for completion or timeout (longer for multiple images)
            timeout = MODEL_CONFIG.get("color_change_timeout", 10) * 3  # 3x normal timeout for multiple tones
            start_time = time.time()
            while not result["completed"]:
                if time.time() - start_time > timeout:
                    raise TimeoutError(f"Color change with tones timed out after {timeout} seconds")
                time.sleep(0.1)  # Small delay to avoid busy waiting
            
            # Check for errors
            if result["error"]:
                raise Exception(result["error"])
            
            return result["images"]
                
        except TimeoutError:
            raise ImageProcessingException(f"Color change with tones timed out after {timeout} seconds. Please try with a smaller image or try again later.")
        except Exception as e:
            logger.error(f"Color change with tones processing failed: {str(e)}")
            raise ImageProcessingException(f"Color change with tones processing failed: {str(e)}")