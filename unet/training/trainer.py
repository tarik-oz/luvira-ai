"""
Training module for hair segmentation U-Net model.
Handles model training with callbacks and logging.
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from pathlib import Path
from typing import Optional, Dict, Any
import logging

from config import (
    TRAINING_CONFIG, CALLBACKS_CONFIG, 
    BEST_MODEL_PATH, LATEST_MODEL_PATH
)
from models.unet_model import create_unet_model
from data.data_loader import create_data_loader

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set TensorFlow logging level
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"


class HairSegmentationTrainer:
    """
    Trainer class for hair segmentation U-Net model.
    
    Handles model training, callbacks, and logging.
    """
    
    def __init__(self, 
                 model_path: Path = BEST_MODEL_PATH,
                 latest_model_path: Path = LATEST_MODEL_PATH,
                 **training_config):
        """
        Initialize the trainer.
        
        Args:
            model_path: Path to save the best model
            latest_model_path: Path to save the latest model
            **training_config: Training configuration parameters
        """
        self.model_path = Path(model_path)
        self.latest_model_path = Path(latest_model_path)
        
        # Create model directory if it doesn't exist
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Training configuration
        self.config = {**TRAINING_CONFIG, **training_config}
        
        # Model and data
        self.model = None
        self.data_loader = None
        self.history = None
        
    def setup_model(self, input_shape: tuple = None) -> tf.keras.Model:
        """
        Setup the U-Net model.
        
        Args:
            input_shape: Input shape for the model
            
        Returns:
            Compiled U-Net model
        """
        logger.info("Setting up U-Net model...")
        
        # Create model
        self.model = create_unet_model(input_shape=input_shape)
        
        # Compile model
        self.model.compile_model(
            optimizer=self.config["optimizer"],
            loss=self.config["loss_function"],
            metrics=self.config["metrics"]
        )
        
        logger.info("Model setup complete")
        self.model.summary()
        
        return self.model.model
    
    def setup_data(self, load_processed: bool = True) -> tuple:
        """
        Setup training and validation data.
        
        Args:
            load_processed: Whether to load processed data or process raw data
            
        Returns:
            Tuple of (train_images, train_masks, val_images, val_masks)
        """
        logger.info("Setting up data...")
        
        self.data_loader = create_data_loader()
        
        if load_processed:
            try:
                data = self.data_loader.load_processed_data()
                logger.info("Loaded processed data successfully")
            except FileNotFoundError:
                logger.warning("Processed data not found. Processing raw data...")
                data = self._process_raw_data()
        else:
            data = self._process_raw_data()
        
        return data
    
    def _process_raw_data(self) -> tuple:
        """
        Process raw data from images and masks directories.
        
        Returns:
            Tuple of (train_images, train_masks, val_images, val_masks)
        """
        # Load and process raw data
        self.data_loader.load_data()
        data = self.data_loader.split_data(
            validation_split=self.config["validation_split"],
            random_seed=self.config["random_seed"]
        )
        
        # Save processed data for future use
        self.data_loader.save_processed_data()
        
        return data
    
    def _create_callbacks(self) -> list:
        """
        Create training callbacks.
        
        Returns:
            List of callbacks
        """
        callbacks = []
        
        # Model checkpoint
        checkpoint = ModelCheckpoint(
            filepath=str(self.model_path),
            monitor=CALLBACKS_CONFIG["checkpoint_monitor"],
            save_best_only=CALLBACKS_CONFIG["checkpoint_save_best_only"],
            verbose=1
        )
        callbacks.append(checkpoint)
        
        # Learning rate reduction
        reduce_lr = ReduceLROnPlateau(
            monitor=CALLBACKS_CONFIG["reduce_lr_monitor"],
            patience=CALLBACKS_CONFIG["reduce_lr_patience"],
            factor=CALLBACKS_CONFIG["reduce_lr_factor"],
            min_lr=CALLBACKS_CONFIG["reduce_lr_min_lr"],
            verbose=1
        )
        callbacks.append(reduce_lr)
        
        # Early stopping
        early_stopping = EarlyStopping(
            monitor=CALLBACKS_CONFIG["early_stopping_monitor"],
            patience=CALLBACKS_CONFIG["early_stopping_patience"],
            verbose=1
        )
        callbacks.append(early_stopping)
        
        return callbacks
    
    def train(self, 
              train_images: np.ndarray,
              train_masks: np.ndarray,
              val_images: np.ndarray,
              val_masks: np.ndarray,
              **kwargs) -> tf.keras.callbacks.History:
        """
        Train the model.
        
        Args:
            train_images: Training images
            train_masks: Training masks
            val_images: Validation images
            val_masks: Validation masks
            **kwargs: Additional training parameters
            
        Returns:
            Training history
        """
        if self.model is None:
            raise ValueError("Model not setup. Call setup_model() first.")
        
        logger.info("Starting training...")
        logger.info(f"Training samples: {len(train_images)}")
        logger.info(f"Validation samples: {len(val_images)}")
        
        # Calculate steps per epoch
        batch_size = kwargs.get('batch_size', self.config['batch_size'])
        steps_per_epoch = np.ceil(len(train_images) / batch_size)
        validation_steps = np.ceil(len(val_images) / batch_size)
        
        logger.info(f"Steps per epoch: {steps_per_epoch}")
        logger.info(f"Validation steps: {validation_steps}")
        
        # Create callbacks
        callbacks = self._create_callbacks()
        
        # Training parameters
        training_params = {
            'batch_size': batch_size,
            'epochs': kwargs.get('epochs', self.config['epochs']),
            'verbose': 1,
            'validation_data': (val_images, val_masks),
            'validation_steps': validation_steps,
            'steps_per_epoch': steps_per_epoch,
            'shuffle': True,
            'callbacks': callbacks
        }
        
        # Start training
        self.history = self.model.model.fit(
            train_images,
            train_masks,
            **training_params
        )
        
        # Save latest model
        self.model.model.save(str(self.latest_model_path))
        logger.info(f"Training completed. Best model saved to {self.model_path}")
        
        return self.history
    
    def get_training_summary(self) -> Dict[str, Any]:
        """
        Get training summary information.
        
        Returns:
            Dictionary containing training summary
        """
        if self.history is None:
            return {"status": "No training history available"}
        
        # Get final metrics
        final_train_loss = self.history.history['loss'][-1]
        final_train_acc = self.history.history['accuracy'][-1]
        final_val_loss = self.history.history['val_loss'][-1]
        final_val_acc = self.history.history['val_accuracy'][-1]
        
        # Get best metrics
        best_val_loss = min(self.history.history['val_loss'])
        best_val_acc = max(self.history.history['val_accuracy'])
        
        summary = {
            "epochs_trained": len(self.history.history['loss']),
            "final_train_loss": final_train_loss,
            "final_train_accuracy": final_train_acc,
            "final_val_loss": final_val_loss,
            "final_val_accuracy": final_val_acc,
            "best_val_loss": best_val_loss,
            "best_val_accuracy": best_val_acc,
            "model_path": str(self.model_path),
            "latest_model_path": str(self.latest_model_path)
        }
        
        return summary
    
    def load_trained_model(self, model_path: Optional[Path] = None) -> tf.keras.Model:
        """
        Load a trained model.
        
        Args:
            model_path: Path to the model file
            
        Returns:
            Loaded model
        """
        if model_path is None:
            model_path = self.model_path
        
        if not Path(model_path).exists():
            raise FileNotFoundError(f"Model not found at {model_path}")
        
        logger.info(f"Loading model from {model_path}")
        self.model = tf.keras.models.load_model(str(model_path))
        
        return self.model


def create_trainer(**kwargs) -> HairSegmentationTrainer:
    """
    Factory function to create a trainer.
    
    Args:
        **kwargs: Arguments to pass to HairSegmentationTrainer
        
    Returns:
        HairSegmentationTrainer instance
    """
    return HairSegmentationTrainer(**kwargs) 