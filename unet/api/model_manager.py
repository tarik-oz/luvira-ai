"""
Model Manager for FastAPI - Singleton pattern to load model once and reuse
"""

import torch
import json
from pathlib import Path
from typing import Optional
import logging
from training.trainer import create_trainer
from inference.predictor import create_predictor

logger = logging.getLogger(__name__)


class ModelManager:
    """
    Singleton model manager for FastAPI.
    Loads model once and reuses it for all requests.
    """
    
    _instance = None
    _model = None
    _predictor = None
    _model_path = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ModelManager, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        # Only initialize once
        if not hasattr(self, '_initialized'):
            self._initialized = True
            self._model = None
            self._predictor = None
            self._model_path = None
    
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
            if self._model is not None and self._model_path == model_path:
                logger.info("Model already loaded, skipping...")
                return True
            
            logger.info(f"Loading model from: {model_path}")
            
            # Create trainer and load model
            trainer = create_trainer()
            self._model = trainer.load_trained_model(model_path)
            self._model_path = model_path
            
            # Create predictor
            self._predictor = create_predictor(self._model)
            
            logger.info("Model loaded successfully!")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False
    
    def get_predictor(self):
        """
        Get the predictor instance.
        
        Returns:
            Predictor instance or None if model not loaded
        """
        if self._predictor is None:
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
    
    def get_model_info(self) -> dict:
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
        
        # Try to get config info
        if self._model_path:
            config_path = self._model_path.parent / "config.json"
            if config_path.exists():
                try:
                    with open(config_path, 'r') as f:
                        config_data = json.load(f)
                    info["config"] = config_data
                except Exception as e:
                    logger.warning(f"Could not load config: {e}")
        
        return info


# Global model manager instance
model_manager = ModelManager() 