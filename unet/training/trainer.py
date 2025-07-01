"""
Training module for hair segmentation U-Net model.
Handles model training with callbacks and logging.
"""

import os
import json
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
import logging
from tqdm import tqdm

from config import (
    TRAINING_CONFIG, CALLBACKS_CONFIG, 
    TRAINED_MODELS_DIR, MODEL_CONFIG, MODEL_DIR
)
from models.unet_model import create_unet_model
from data.data_loader import create_data_loader
from utils.model_saving import (
    create_timestamped_folder, save_config_json, 
    save_training_log, save_models_to_folder
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EarlyStopping:
    """Early stopping callback for PyTorch training."""
    
    def __init__(self, patience: int = 7, min_delta: float = 0, monitor: str = 'val_loss'):
        self.patience = patience
        self.min_delta = min_delta
        self.monitor = monitor
        self.counter = 0
        self.best_loss = float('inf')
        
    def __call__(self, val_loss: float) -> bool:
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            
        return self.counter >= self.patience


class HairSegmentationTrainer:
    """
    Trainer class for hair segmentation U-Net model.
    
    Handles model training, callbacks, and logging.
    """
    
    def __init__(self, 
                 **training_config):
        """
        Initialize the trainer.
        
        Args:
            **training_config: Training configuration parameters
        """
        # Create temporary model paths for training
        self.model_path = MODEL_DIR / "best_model.pth"
        self.latest_model_path = MODEL_DIR / "latest_model.pth"
        
        # Create model directory if it doesn't exist
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Training configuration
        self.config = {**TRAINING_CONFIG, **training_config}
        
        # Model and data
        self.model = None
        self.data_loader = None
        self.device = None
        self.optimizer = None
        self.criterion = None
        self.history = {
            'train_loss': [],
            'train_accuracy': [],
            'val_loss': [],
            'val_accuracy': [],
            'train_dice': [],
            'val_dice': [],
            'train_mse': [],
            'val_mse': []
        }
        
        # Setup device
        self._setup_device()
        
    def _setup_device(self):
        """Setup device (CPU/GPU) for training."""
        if self.config["device"] == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif self.config["device"] == "cuda":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device("cpu")
            
        logger.info(f"Using device: {self.device}")
        
    def setup_model(self, input_shape: tuple = None) -> nn.Module:
        """
        Setup the U-Net model.
        
        Args:
            input_shape: Input shape for the model
            
        Returns:
            U-Net model
        """
        logger.info("Setting up U-Net model...")
        
        # Create model
        self.model = create_unet_model(input_shape=input_shape)
        self.model.to(self.device)
        
        # Setup optimizer
        if self.config["optimizer"] == "adam":
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.config["learning_rate"])
        elif self.config["optimizer"] == "sgd":
            self.optimizer = optim.SGD(self.model.parameters(), lr=self.config["learning_rate"])
        else:
            raise ValueError(f"Unsupported optimizer: {self.config['optimizer']}")
        
        # Setup loss function
        if self.config["loss_function"] == "bce":
            self.criterion = nn.BCELoss()
        elif self.config["loss_function"] == "bce_with_logits":
            self.criterion = nn.BCEWithLogitsLoss()
        else:
            raise ValueError(f"Unsupported loss function: {self.config['loss_function']}")
        
        logger.info("Model setup complete")
        self.model.summary()
        
        return self.model
    
    def setup_data(self, load_processed: bool = True) -> Tuple[DataLoader, DataLoader]:
        """
        Setup training and validation data.
        
        Args:
            load_processed: Whether to load processed data or process raw data
            
        Returns:
            Tuple of (train_loader, val_loader)
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
        
        # Log dataset information
        data_info = self.data_loader.get_data_info()
        logger.info(f"Dataset info: {data_info}")
        
        # Convert to PyTorch tensors and create DataLoaders
        train_images, train_masks, val_images, val_masks = data
        
        # Convert to PyTorch format (NCHW)
        train_images = torch.FloatTensor(train_images).permute(0, 3, 1, 2)
        train_masks = torch.FloatTensor(train_masks).unsqueeze(1)  # Add channel dimension
        val_images = torch.FloatTensor(val_images).permute(0, 3, 1, 2)
        val_masks = torch.FloatTensor(val_masks).unsqueeze(1)  # Add channel dimension
        
        # Create datasets
        train_dataset = TensorDataset(train_images, train_masks)
        val_dataset = TensorDataset(val_images, val_masks)
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset, 
            batch_size=self.config["batch_size"], 
            shuffle=True,
            num_workers=0  # Set to 0 for Windows compatibility
        )
        val_loader = DataLoader(
            val_dataset, 
            batch_size=self.config["batch_size"], 
            shuffle=False,
            num_workers=0  # Set to 0 for Windows compatibility
        )
        
        return train_loader, val_loader
    
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
    
    def _calculate_accuracy(self, predictions: torch.Tensor, targets: torch.Tensor) -> float:
        """Calculate accuracy for binary segmentation."""
        with torch.no_grad():
            # Convert predictions and targets to binary
            binary_preds = (predictions > 0.5).float()
            binary_targets = (targets > 0.5).float()
            # Calculate accuracy
            accuracy = (binary_preds == binary_targets).float().mean().item()
        return accuracy
    
    def _calculate_dice(self, predictions: torch.Tensor, targets: torch.Tensor) -> float:
        """Calculate Dice coefficient for probability masks."""
        with torch.no_grad():
            smooth = 1e-6
            preds = predictions.view(-1)
            targs = targets.view(-1)
            intersection = (preds * targs).sum()
            dice = (2. * intersection + smooth) / (preds.sum() + targs.sum() + smooth)
        return dice.item()

    def _calculate_mse(self, predictions: torch.Tensor, targets: torch.Tensor) -> float:
        """Calculate Mean Squared Error for probability masks."""
        with torch.no_grad():
            mse = torch.mean((predictions - targets) ** 2)
        return mse.item()
    
    def _train_epoch(self, train_loader: DataLoader) -> Tuple[float, float, float, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        total_accuracy = 0.0
        total_dice = 0.0
        total_mse = 0.0
        num_batches = 0
        
        for batch_idx, (images, masks) in enumerate(tqdm(train_loader, desc="Training")):
            images = images.to(self.device)
            masks = masks.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            predictions = self.model(images)
            loss = self.criterion(predictions, masks)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Calculate accuracy
            accuracy = self._calculate_accuracy(predictions, masks)
            dice = self._calculate_dice(predictions, masks)
            mse = self._calculate_mse(predictions, masks)
            
            total_loss += loss.item()
            total_accuracy += accuracy
            total_dice += dice
            total_mse += mse
            num_batches += 1
        
        return (total_loss / num_batches, total_accuracy / num_batches, total_dice / num_batches, total_mse / num_batches)
    
    def _validate_epoch(self, val_loader: DataLoader) -> Tuple[float, float, float, float]:
        """Validate for one epoch."""
        self.model.eval()
        total_loss = 0.0
        total_accuracy = 0.0
        total_dice = 0.0
        total_mse = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch_idx, (images, masks) in enumerate(tqdm(val_loader, desc="Validation")):
                images = images.to(self.device)
                masks = masks.to(self.device)
                
                # Forward pass
                predictions = self.model(images)
                loss = self.criterion(predictions, masks)
                
                # Calculate accuracy
                accuracy = self._calculate_accuracy(predictions, masks)
                dice = self._calculate_dice(predictions, masks)
                mse = self._calculate_mse(predictions, masks)
                
                total_loss += loss.item()
                total_accuracy += accuracy
                total_dice += dice
                total_mse += mse
                num_batches += 1
        
        return (total_loss / num_batches, total_accuracy / num_batches, total_dice / num_batches, total_mse / num_batches)
    
    def train(self, 
              train_loader: DataLoader,
              val_loader: DataLoader,
              **kwargs) -> Dict[str, list]:
        """
        Train the model.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            **kwargs: Additional training parameters
            
        Returns:
            Training history
        """
        if self.model is None:
            raise ValueError("Model not setup. Call setup_model() first.")
        
        epochs = kwargs.get('epochs', self.config['epochs'])
        
        logger.info("Starting training...")
        logger.info(f"Training samples: {len(train_loader.dataset)}")
        logger.info(f"Validation samples: {len(val_loader.dataset)}")
        logger.info(f"Epochs: {epochs}")
        
        # Setup callbacks
        early_stopping = EarlyStopping(
            patience=CALLBACKS_CONFIG["early_stopping_patience"],
            monitor=CALLBACKS_CONFIG["early_stopping_monitor"]
        )
        
        best_val_loss = float('inf')
        
        for epoch in range(epochs):
            logger.info(f"\nEpoch {epoch+1}/{epochs}")
            
            # Training
            train_loss, train_acc, train_dice, train_mse = self._train_epoch(train_loader)
            
            # Validation
            val_loss, val_acc, val_dice, val_mse = self._validate_epoch(val_loader)
            
            # Log results
            logger.info(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Train Dice: {train_dice:.4f}, Train MSE: {train_mse:.6f}")
            logger.info(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val Dice: {val_dice:.4f}, Val MSE: {val_mse:.6f}")
            
            # Save history
            self.history['train_loss'].append(train_loss)
            self.history['train_accuracy'].append(train_acc)
            self.history['train_dice'].append(train_dice)
            self.history['train_mse'].append(train_mse)
            self.history['val_loss'].append(val_loss)
            self.history['val_accuracy'].append(val_acc)
            self.history['val_dice'].append(val_dice)
            self.history['val_mse'].append(val_mse)
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(self.model.state_dict(), self.model_path)
                logger.info(f"Saved best model with val_loss: {val_loss:.4f}")
            
            # Save latest model
            torch.save(self.model.state_dict(), self.latest_model_path)
            
            # Early stopping
            if early_stopping(val_loss):
                logger.info("Early stopping triggered")
                break
        
        logger.info("Training completed!")
        return self.history
    
    def get_training_summary(self) -> Dict[str, Any]:
        """
        Get training summary.
        
        Returns:
            Dictionary containing training summary
        """
        if not self.history['train_loss']:
            return {}
        
        return {
            'final_train_loss': self.history['train_loss'][-1],
            'final_train_accuracy': self.history['train_accuracy'][-1],
            'final_train_dice': self.history['train_dice'][-1],
            'final_train_mse': self.history['train_mse'][-1],
            'final_val_loss': self.history['val_loss'][-1],
            'final_val_accuracy': self.history['val_accuracy'][-1],
            'final_val_dice': self.history['val_dice'][-1],
            'final_val_mse': self.history['val_mse'][-1],
            'best_val_loss': min(self.history['val_loss']),
            'best_val_accuracy': max(self.history['val_accuracy']),
            'best_val_dice': max(self.history['val_dice']),
            'best_val_mse': min(self.history['val_mse']),
            'total_epochs': len(self.history['train_loss']),
            'device_used': str(self.device)
        }
    
    def load_trained_model(self, model_path: Path) -> nn.Module:
        """
        Load a trained model.
        
        Args:
            model_path: Path to the model file (required)
            
        Returns:
            Loaded model
        """
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        # Try to find config.json in the same folder as the model
        config_path = model_path.parent / "config.json"
        
        if config_path.exists():
            # Load model config from JSON
            try:
                with open(config_path, 'r') as f:
                    config_data = json.load(f)
                
                model_config = config_data.get("model_config", {})
                logger.info(f"Loading model with config from: {config_path}")
                logger.info(f"Model config: {model_config}")
                
                # Create model with the loaded config
                self.model = create_unet_model(**model_config)
                self.model.to(self.device)
                
            except Exception as e:
                logger.warning(f"Failed to load config from {config_path}: {e}")
                logger.info("Falling back to default config")
                # Fallback to default config
                if self.model is None:
                    self.setup_model()
        else:
            # No config.json found, use default config
            logger.info(f"No config.json found at {config_path}, using default config")
            if self.model is None:
                self.setup_model()
        
        # Load model weights
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        
        logger.info(f"Loaded model from {model_path}")
        return self.model

    def save_trained_model(self, dataset_info: Dict[str, Any] = None) -> Path:
        from config import MODEL_CONFIG
        summary = self.get_training_summary()
        if not summary:
            raise ValueError("No training history available. Train the model first.")
        folder_path = create_timestamped_folder(TRAINED_MODELS_DIR, summary['best_val_accuracy'])
        save_config_json(folder_path, MODEL_CONFIG, self.config, dataset_info, summary)
        save_training_log(folder_path, self.device, self.history, summary)
        save_models_to_folder(folder_path, self.model_path, self.latest_model_path)
        
        # Clean up temporary files from models directory
        try:
            if self.model_path.exists():
                self.model_path.unlink()
                logger.info(f"Deleted temporary file: {self.model_path}")
            if self.latest_model_path.exists():
                self.latest_model_path.unlink()
                logger.info(f"Deleted temporary file: {self.latest_model_path}")
        except Exception as e:
            logger.warning(f"Failed to delete temporary files: {e}")
        
        logging.info(f"Model saved successfully to: {folder_path}")
        return folder_path


def create_trainer(**kwargs) -> HairSegmentationTrainer:
    """
    Factory function to create a trainer.
    
    Args:
        **kwargs: Arguments for HairSegmentationTrainer
        
    Returns:
        HairSegmentationTrainer instance
    """
    return HairSegmentationTrainer(**kwargs) 