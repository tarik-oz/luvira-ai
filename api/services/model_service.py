"""
Model management service for Hair Segmentation API
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional
from ..core.exceptions import ModelNotLoadedException, ModelLoadException
from ..utils import ModelPathValidator
from ..config import MODEL_CONFIG
from model.training.trainer import create_trainer
from model.inference.predictor import create_predictor

logger = logging.getLogger(__name__)


class ModelService:
    """Service for handling model management operations"""
    
    def __init__(self):
        self._model = None
        self._predictor = None
        self._model_path = None
        self._trainer = None
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get comprehensive information about the loaded model
        
        Returns:
            Dictionary with model information
        """
        if not self.is_model_loaded():
            return {
                "is_loaded": False,
                "error": "Model not loaded"
            }
        
        info = {
            "model_path": str(self._model_path),
            "model_loaded": True,
            "device": str(next(self._model.parameters()).device),
            "is_loaded": True
        }
        
        # Try to get config info from the model directory
        if self._model_path:
            config_path = self._model_path.parent / "config.json"
            if config_path.exists():
                try:
                    with open(config_path, 'r') as f:
                        config_data = json.load(f)
                    
                    # Extract model config information
                    model_config = config_data.get("model_config", {})
                    info["model_type"] = model_config.get("model_type", "unknown")
                    info["input_shape"] = model_config.get("input_shape", [3, 256, 256])
                    info["output_shape"] = [model_config.get("output_channels", 1), 256, 256]
                    info["config"] = config_data
                    
                    logger.info(f"Loaded model config from {config_path}")
                except Exception as e:
                    logger.warning(f"Could not load config: {e}")
                    # Fallback values
                    info["model_type"] = "unknown"
                    info["input_shape"] = [3, 256, 256]
                    info["output_shape"] = [1, 256, 256]
            else:
                logger.warning(f"Config file not found at {config_path}")
                # Fallback values
                info["model_type"] = "unknown"
                info["input_shape"] = [3, 256, 256]
                info["output_shape"] = [1, 256, 256]
        
        return info
    
    def load_model(self, model_path: Optional[str] = None) -> bool:
        """
        Load model with validation and error handling
        
        Args:
            model_path: Path to model file (optional, uses default if not provided)
            
        Returns:
            True if successful, False otherwise
        """
        try:
            model_path = model_path or MODEL_CONFIG["default_model_path"]
            
            # Validate model path
            ModelPathValidator.validate_model_path(model_path)
            
            # Check if model is already loaded
            if self._is_same_model_loaded(Path(model_path)):
                logger.info("Model already loaded, skipping...")
                return True
            
            # Check device preference
            device_preference = MODEL_CONFIG.get("device_preference", "auto")
            logger.info(f"Loading model with device preference: {device_preference}")
            
            # Set device based on preference
            if device_preference == "cuda":
                logger.info("Forcing CUDA device")
            elif device_preference == "cpu":
                logger.info("Forcing CPU device")
            else:  # auto
                logger.info("Using automatic device selection")
            
            # Load model
            success = self._load_model_internal(model_path, device_preference)
            
            if success:
                logger.info(f"Model loaded successfully from {model_path}")
                return True
            else:
                logger.error(f"Failed to load model from {model_path}")
                return False
                
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise ModelLoadException(model_path, str(e))
    
    def reload_model(self, model_path: Optional[str] = None) -> bool:
        """
        Force reload model with validation
        
        Args:
            model_path: Path to model file (optional, uses default if not provided)
            
        Returns:
            True if successful, False otherwise
        """
        try:
            model_path = model_path or MODEL_CONFIG["default_model_path"]
            
            # Validate model path
            ModelPathValidator.validate_model_path(model_path)
            
            logger.info(f"Reloading model from {model_path}")
            
            # Backup current model state
            old_model = self._model
            old_predictor = self._predictor
            old_model_path = self._model_path
            old_trainer = self._trainer
            
            # Clear current model
            self.clear_model()
            
            # Try to load new model
            device_preference = MODEL_CONFIG.get("device_preference", "auto")
            success = self._load_model_internal(model_path, device_preference)
            if success:
                return True
            else:
                # Restore old model if new model failed to load
                logger.warning(f"Failed to load new model from {model_path}, restoring old model")
                self._model = old_model
                self._predictor = old_predictor
                self._model_path = old_model_path
                self._trainer = old_trainer
                return False
                
        except Exception as e:
            logger.error(f"Error reloading model: {e}")
            raise ModelLoadException(model_path, str(e))
    
    def clear_model(self) -> None:
        """
        Clear model from memory with logging
        """
        try:
            logger.info("Clearing model from memory")
            self._model = None
            self._predictor = None
            self._model_path = None
            self._trainer = None
            logger.info("Model cleared successfully")
        except Exception as e:
            logger.error(f"Error clearing model: {e}")
            raise ModelNotLoadedException(f"Error clearing model: {str(e)}")
    
    def is_model_loaded(self) -> bool:
        """
        Check if model is loaded
        
        Returns:
            True if model is loaded, False otherwise
        """
        return self._predictor is not None
    
    def get_model_path(self) -> Optional[Path]:
        """
        Get current model path
        
        Returns:
            Path to loaded model or None if not loaded
        """
        return self._model_path
    
    def get_predictor(self):
        """
        Get predictor instance with validation
        
        Returns:
            Predictor instance
            
        Raises:
            ModelNotLoadedException: If model is not loaded
        """
        if not self.is_model_loaded():
            raise ModelNotLoadedException("Model is not loaded")
        
        if self._predictor is None:
            raise ModelNotLoadedException("Predictor not available")
        
        return self._predictor
    
    def _is_same_model_loaded(self, model_path: Path) -> bool:
        """Check if the same model is already loaded."""
        return (self._model is not None and 
                self._model_path is not None and 
                self._model_path == model_path)
    
    def _load_model_internal(self, model_path: str, device_preference: str = "auto") -> bool:
        """
        Internal method to load the model
        
        Args:
            model_path: Path to the model file
            device_preference: Device preference (cuda, cpu, auto)
            
        Returns:
            True if successful, False otherwise
        """
        try:
            model_path = Path(model_path)
            
            logger.info(f"Loading model from: {model_path}")
            logger.info(f"Device preference: {device_preference}")
            
            # Create trainer and load model
            self._trainer = create_trainer()
            self._model, _ = self._trainer.load_trained_model(model_path)
            self._model_path = model_path
            
            # Create predictor
            self._predictor = create_predictor(self._model)
            
            # Log device information if available
            if hasattr(self._predictor, 'device'):
                logger.info(f"Model loaded on device: {self._predictor.device}")
            else:
                logger.info("Model loaded (device info not available)")
            
            logger.info("Model loaded successfully!")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False 