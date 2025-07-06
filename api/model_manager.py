"""
Model Manager for FastAPI - Singleton pattern to load model once and reuse
"""

import json
from pathlib import Path
from typing import Dict, Any, Optional
import logging

from model.training.trainer import create_trainer
from model.inference.predictor import create_predictor

logger = logging.getLogger(__name__)


class ModelManager:
    """
    Singleton model manager for FastAPI.
    Loads model once and reuses it for all requests.
    """
    
    _instance: Optional['ModelManager'] = None
    _initialized: bool = False
    
    def __new__(cls) -> 'ModelManager':
        if cls._instance is None:
            cls._instance = super(ModelManager, cls).__new__(cls)
        return cls._instance
    
    def __init__(self) -> None:
        # Only initialize once
        if not self._initialized:
            self._model = None
            self._predictor = None
            self._model_path = None
            self._trainer = None
            self._initialized = True
    
    def load_model(self, model_path: str) -> bool:
        """
        Load the model once.
        
        Args:
            model_path: Path to the model file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            model_path = Path(model_path)
            
            # Check if model is already loaded
            if self._is_same_model_loaded(model_path):
                logger.info("Model already loaded, skipping...")
                return True
            
            logger.info(f"Loading model from: {model_path}")
            
            # Create trainer and load model
            self._trainer = create_trainer()
            self._model, _ = self._trainer.load_trained_model(model_path)
            self._model_path = model_path
            
            # Create predictor
            self._predictor = create_predictor(self._model)
            
            logger.info("Model loaded successfully!")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False
    
    def _is_same_model_loaded(self, model_path: Path) -> bool:
        """Check if the same model is already loaded."""
        return (self._model is not None and 
                self._model_path is not None and 
                self._model_path == model_path)
    
    def get_predictor(self) -> Optional[Any]:
        """
        Get the predictor instance.
        
        Returns:
            Predictor instance or None if model not loaded
        """
        if not self.is_model_loaded():
            logger.error("Model not loaded. Call load_model() first.")
            return None
        return self._predictor
    
    def is_model_loaded(self) -> bool:
        """
        Check if model is loaded.
        
        Returns:
            True if model is loaded, False otherwise
        """
        return self._predictor is not None
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the loaded model.
        
        Returns:
            Dictionary with model information
        """
        if not self.is_model_loaded():
            return {"error": "Model not loaded"}
        
        info = {
            "model_path": str(self._model_path),
            "model_loaded": True,
            "device": str(next(self._model.parameters()).device)
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
                except Exception as e:
                    logger.warning(f"Could not load config: {e}")
                    # Fallback values
                    info["model_type"] = "unknown"
                    info["input_shape"] = [3, 256, 256]
                    info["output_shape"] = [1, 256, 256]
        
        return info
    
    def reload_model(self, model_path: str) -> bool:
        """
        Force reload the model even if it's the same path.
        
        Args:
            model_path: Path to the model file
            
        Returns:
            True if successful, False otherwise
        """
        # Backup current model state
        old_model = self._model
        old_predictor = self._predictor
        old_model_path = self._model_path
        old_trainer = self._trainer
        
        # Clear current model
        self.clear_model()
        
        # Try to load new model
        success = self.load_model(model_path)
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
    
    def clear_model(self) -> None:
        """
        Clear the loaded model from memory.
        """
        self._model = None
        self._predictor = None
        self._model_path = None
        self._trainer = None
        logger.info("Model cleared from memory")
    
    def get_model_path(self) -> Optional[Path]:
        """
        Get the current model path.
        
        Returns:
            Path to the loaded model or None if not loaded
        """
        return self._model_path


# Global model manager instance
model_manager = ModelManager() 