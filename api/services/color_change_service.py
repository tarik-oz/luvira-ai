"""
Color change service for hair color modification
"""

import logging
import signal
import numpy as np
import sys
import os

# Add color_changer module to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'color_changer'))
from color_changer import ColorTransformer

from ..core.exceptions import PredictionException, ImageProcessingException
from ..utils import FileValidator, ColorValidator, ImageUtils
from ..config import MODEL_CONFIG
from .session_manager import session_manager

logger = logging.getLogger(__name__)


class TimeoutException(Exception):
    """Exception raised when operation times out"""
    pass


def timeout_handler(signum, frame):
    """Signal handler for timeout"""
    raise TimeoutException("Operation timed out")


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
            List of dicts: { name: str, rgb: [R, G, B] }
        """
        # Import here to avoid circular import
        from color_changer.config.color_config import COLORS
        return [
            {"name": name, "rgb": rgb}
            for rgb, name in COLORS
        ]
    
    @staticmethod
    def get_available_tones(color_name: str) -> list:
        """
        Get list of available tones for a specific color (case-insensitive)
        
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
    
    def upload_and_prepare_image(self, file) -> str:
        """
        Upload image, validate it, generate mask and cache everything
        
        Args:
            file: Uploaded image file
            
        Returns:
            Session ID for subsequent operations
        """
        # Validate file
        FileValidator.validate_upload_file(file)
        
        # Check if model is loaded
        if not self.prediction_service.model_service.is_model_loaded():
            raise PredictionException("Model is not loaded - cannot generate hair mask")
        
        try:
            # Read and process image
            original_image = ImageUtils.process_uploaded_file(file)
            
            # Generate mask
            mask_bytes = self.prediction_service.predict_mask_file(file)
            mask_array = ImageUtils.bytes_to_mask(mask_bytes)
            
            # Create session and save data
            session_id = session_manager.create_session()
            session_manager.save_session_data(session_id, original_image, mask_array)
            
            logger.info(f"Image uploaded and prepared successfully. Session: {session_id}")
            return session_id
            
        except Exception as e:
            logger.error(f"Failed to upload and prepare image: {str(e)}")
            raise ImageProcessingException(f"Failed to upload and prepare image: {str(e)}")
    
    def change_hair_color_with_session(self, session_id: str, color_name: str, tone: str = None) -> bytes:
        """
        Change hair color using cached session data (FAST!)
        
        Args:
            session_id: Session identifier
            color_name: Color name from COLORS config
            tone: Optional tone name
            
        Returns:
            Color-changed image bytes
        """
        # Validate inputs
        ColorValidator.validate_color_name(color_name)
        correct_color_name = ColorValidator.get_correct_color_name(color_name)
        
        if tone:
            ColorValidator.validate_tone_name(color_name, tone)
            correct_tone = ColorValidator.get_correct_tone_name(color_name, tone)
        else:
            correct_tone = None
        
        try:
            # Load cached data (NO MASK GENERATION!)
            session_data = session_manager.load_session_data(session_id)
            original_image = session_data['image']
            mask_array = session_data['mask']
            
            # Apply color change
            if correct_tone:
                result_image = self._apply_color_change_with_tone(
                    original_image, mask_array, correct_color_name, correct_tone
                )
            else:
                result_image = self._apply_color_change(
                    original_image, mask_array, correct_color_name
                )
            
            # Convert result to bytes
            result_bytes = ImageUtils.image_to_bytes(result_image, is_rgb=True)
            
            logger.info(f"Successfully changed hair color to {correct_color_name}{f' ({correct_tone})' if correct_tone else ''} for session: {session_id}")
            return result_bytes
            
        except Exception as e:
            logger.error(f"Color change failed for session {session_id}: {str(e)}")
            raise ImageProcessingException(f"Color change failed: {str(e)}")
    
    def change_hair_color(self, file, color_name: str, tone: str = None) -> bytes:
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
            # Read and process image
            original_image = ImageUtils.process_uploaded_file(file)
            
            # Generate hair mask using prediction service with timeout handling
            try:
                mask_bytes = self.prediction_service.predict_mask_file(file)
            except Exception as e:
                if "timed out" in str(e).lower():
                    raise ImageProcessingException("Mask generation timed out. Please try with a smaller image or try again later.")
                else:
                    raise ImageProcessingException(f"Mask generation failed: {str(e)}")
            
            # Convert mask bytes to numpy array
            mask_array = ImageUtils.bytes_to_mask(mask_bytes)
            
            # Apply color change
            if correct_tone:
                result_image = self._apply_color_change_with_tone(
                    original_image, mask_array, correct_color_name, correct_tone
                )
            else:
                result_image = self._apply_color_change(
                    original_image, mask_array, correct_color_name
                )
            
            # Convert result to bytes
            result_bytes = ImageUtils.image_to_bytes(result_image, is_rgb=True)
            
            logger.info(f"Successfully changed hair color to {correct_color_name}{f' ({correct_tone})' if correct_tone else ''} for file: {file.filename}")
            return result_bytes
            
        except Exception as e:
            logger.error(f"Color change failed for file {file.filename}: {str(e)}")
            raise ImageProcessingException(f"Color change failed: {str(e)}")
        finally:
            # Cleanup
            ImageUtils.cleanup_temp_file(temp_input_path)
    
    def _apply_color_change(self, original_image: np.ndarray, mask_array: np.ndarray, color_name: str) -> np.ndarray:
        """Apply color change by name with timeout"""
        timeout = MODEL_CONFIG.get("color_change_timeout", 30)
        
        # Set up timeout signal (Unix systems only)
        if hasattr(signal, 'SIGALRM'):
            old_handler = signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(timeout)
        
        try:
            result_image = self.color_transformer.change_hair_color(original_image, mask_array, color_name)
            return result_image
        except TimeoutException:
            raise ImageProcessingException(f"Color change timed out after {timeout} seconds. Please try with a smaller image.")
        except Exception as e:
            logger.error(f"Color change processing failed: {str(e)}")
            raise ImageProcessingException(f"Color change processing failed: {str(e)}")
        finally:
            # Restore original signal handler
            if hasattr(signal, 'SIGALRM'):
                signal.alarm(0)  # Cancel the alarm
                signal.signal(signal.SIGALRM, old_handler)
    
    def _apply_color_change_with_tone(self, original_image: np.ndarray, mask_array: np.ndarray, color_name: str, tone: str) -> np.ndarray:
        """Apply color change with specific tone and timeout"""
        timeout = MODEL_CONFIG.get("color_change_timeout", 30)
        
        # Set up timeout signal (Unix systems only)
        if hasattr(signal, 'SIGALRM'):
            old_handler = signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(timeout)
        
        try:
            result_image = self.color_transformer.apply_color_with_tone(original_image, mask_array, color_name, tone)
            return result_image
        except TimeoutException:
            raise ImageProcessingException(f"Color change timed out after {timeout} seconds. Please try with a smaller image.")
        except Exception as e:
            logger.error(f"Color change processing failed: {str(e)}")
            raise ImageProcessingException(f"Color change processing failed: {str(e)}")
        finally:
            # Restore original signal handler
            if hasattr(signal, 'SIGALRM'):
                signal.alarm(0)  # Cancel the alarm
                signal.signal(signal.SIGALRM, old_handler) 