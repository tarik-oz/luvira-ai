"""
Color change service for hair color modification
"""

import logging
import cv2
import numpy as np
import sys
import os
import concurrent.futures as _futures

# Add color_changer module to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'color_changer'))
from color_changer import ColorTransformer

from ..core.exceptions import (
    APIException,
    PredictionException,
    ImageProcessingException,
    NoHairDetectedException,
)
from ..utils import FileValidator, ColorValidator, ImageUtils
from ..config import MODEL_CONFIG
from .session_manager import session_manager

logger = logging.getLogger(__name__)


class TimeoutException(Exception):
    """Exception raised when operation times out"""
    pass


class ColorChangeService:
    """Service for handling hair color change operations"""
    
    def __init__(self, prediction_service):
        self.prediction_service = prediction_service
        self.color_transformer = ColorTransformer()
    
    def _run_with_timeout(self, func, timeout_seconds: int, *args, **kwargs):
        """Run blocking function with a timeout in a worker thread (thread-safe)."""
        if not timeout_seconds or timeout_seconds <= 0:
            return func(*args, **kwargs)
        with _futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(func, *args, **kwargs)
            try:
                return future.result(timeout=timeout_seconds)
            except _futures.TimeoutError:
                raise TimeoutException(f"Operation timed out after {timeout_seconds} seconds")
    
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

            # Validate hair presence in mask (upload-only gate; does not affect actual processing mask)
            try:
                pixel_threshold = int(MODEL_CONFIG.get("hair_presence_pixel_threshold", 64))
                hair_pixels = int((mask_array > pixel_threshold).sum())
                total_pixels = int(mask_array.size)
                hair_ratio = (hair_pixels / max(total_pixels, 1)) if total_pixels else 0.0
            except Exception:
                hair_ratio = 0.0
            minimal_ratio = MODEL_CONFIG.get("minimal_hair_ratio", 0.005)  # 0.5%
            if hair_ratio < minimal_ratio:
                raise NoHairDetectedException("No hair area detected in the image")
            
            # Create session and save data
            session_id = session_manager.create_session()
            session_manager.save_session_data(session_id, original_image, mask_array)
            
            logger.info(f"Image uploaded and prepared successfully. Session: {session_id}")
            return session_id
            
        except NoHairDetectedException:
            # Preserve specific error for frontend mapping
            raise
        except APIException:
            # Bubble up structured API exceptions
            raise
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

    def iter_overlays_with_session(self, session_id: str, color_name: str, webp_quality: int = 90):
        """
        Generator that yields (part_name, webp_bytes) for base and each tone.

        Yields:
            ("base", bytes) first, followed by (tone_name, bytes) for each tone.
        """
        # Validate color
        ColorValidator.validate_color_name(color_name)
        correct_color_name = ColorValidator.get_correct_color_name(color_name)

        # Load session data (raises SessionExpiredException if invalid)
        session_data = session_manager.load_session_data(session_id)
        original_image = session_data['image']
        mask_array = session_data['mask']

        def encode_webp_rgba(img_rgb: np.ndarray, alpha_mask: np.ndarray, q: int) -> bytes:
            # Encode full composited RGB (opaque) to ensure exact visual parity with CLI previews
            # This avoids double blending and color fringing at semi-transparent edges in the browser.
            bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
            params = [cv2.IMWRITE_WEBP_QUALITY, int(q)]
            ok, buf = cv2.imencode('.webp', bgr, params)
            if not ok:
                raise ImageProcessingException("WEBP encode failed")
            return buf.tobytes()

        # 1) Base overlay (yield early)
        try:
            base_rgb = self._apply_color_change(original_image, mask_array, correct_color_name)
            base_webp = encode_webp_rgba(base_rgb, mask_array, webp_quality)
            yield ("base", base_webp)
        except Exception as e:
            logger.error(f"Failed to generate base overlay for session {session_id}: {e}")
            raise

        # 2) Tones (yield incrementally)
        try:
            tone_names = self.get_available_tones(correct_color_name)
            for tone_name in tone_names:
                try:
                    tone_rgb = self._apply_color_change_with_tone(
                        original_image, mask_array, correct_color_name, tone_name
                    )
                    tone_webp = encode_webp_rgba(tone_rgb, mask_array, webp_quality)
                    yield (tone_name, tone_webp)
                except Exception as tone_err:
                    logger.warning(
                        f"Tone generation failed for session {session_id}, color {correct_color_name}, tone {tone_name}: {tone_err}"
                    )
                    continue
        except Exception as e:
            logger.error(f"Failed to generate tones for session {session_id}: {e}")
            # Stop iteration; base already sent
            return
    
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
        """Apply color change by name with timeout (thread-safe)."""
        timeout = MODEL_CONFIG.get("color_change_timeout", 30)
        try:
            return self._run_with_timeout(
                self.color_transformer.change_hair_color,
                timeout,
                original_image,
                mask_array,
                color_name,
            )
        except TimeoutException:
            raise ImageProcessingException(f"Color change timed out after {timeout} seconds. Please try with a smaller image.")
        except Exception as e:
            logger.error(f"Color change processing failed: {str(e)}")
            raise ImageProcessingException(f"Color change processing failed: {str(e)}")
    
    def _apply_color_change_with_tone(self, original_image: np.ndarray, mask_array: np.ndarray, color_name: str, tone: str) -> np.ndarray:
        """Apply color change with specific tone and timeout (thread-safe)."""
        timeout = MODEL_CONFIG.get("color_change_timeout", 30)
        try:
            return self._run_with_timeout(
                self.color_transformer.apply_color_with_tone,
                timeout,
                original_image,
                mask_array,
                color_name,
                tone,
            )
        except TimeoutException:
            raise ImageProcessingException(f"Color change timed out after {timeout} seconds. Please try with a smaller image.")
        except Exception as e:
            logger.error(f"Color change processing failed: {str(e)}")
            raise ImageProcessingException(f"Color change processing failed: {str(e)}")