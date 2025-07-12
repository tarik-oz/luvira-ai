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
import torch.nn.functional as F

try:
    from ..config import (
        TRAINING_CONFIG, CALLBACKS_CONFIG, 
        TRAINED_MODELS_DIR, MODEL_CONFIG, MODEL_DIR
    )
    from ..models.unet_model import create_unet_model
    from ..models.attention_unet_model import create_attention_unet_model
    from ..data.data_loader import create_data_loader
    from ..utils.model_saving import (
        create_timestamped_folder, save_config_json, 
        save_training_log, save_models_to_folder
    )
except ImportError:
    from config import (
        TRAINING_CONFIG, CALLBACKS_CONFIG, 
        TRAINED_MODELS_DIR, MODEL_CONFIG, MODEL_DIR
    )
    from models.unet_model import create_unet_model
    from models.attention_unet_model import create_attention_unet_model
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
        self.best_metric = float('inf') if monitor.endswith('loss') else float('-inf')
        
    def __call__(self, metric_value: float) -> bool:
        # For loss metrics: lower is better
        # For accuracy/dice metrics: higher is better
        if self.monitor.endswith('loss'):
            is_better = metric_value < self.best_metric - self.min_delta
        else:
            is_better = metric_value > self.best_metric + self.min_delta
            
        if is_better:
            self.best_metric = metric_value
            self.counter = 0
        else:
            self.counter += 1
            
        return self.counter >= self.patience


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.8, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.bce = nn.BCELoss(reduction='none')

    def forward(self, inputs, targets):
        bce_loss = self.bce(inputs, targets)
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        return focal_loss.mean()


class DiceLoss(nn.Module):
    def __init__(self, smooth=1.):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, inputs, targets):
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        intersection = (inputs * targets).sum()
        dice = (2. * intersection + self.smooth) / (inputs.sum() + targets.sum() + self.smooth)
        return 1 - dice


class ComboLoss(nn.Module):
    def __init__(self, bce_weight=0.5, dice_weight=0.5):
        super(ComboLoss, self).__init__()
        self.bce = nn.BCELoss()
        self.dice = DiceLoss()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight

    def forward(self, inputs, targets):
        return self.bce_weight * self.bce(inputs, targets) + self.dice_weight * self.dice(inputs, targets)


class BoundaryLoss(nn.Module):
    def __init__(self):
        super(BoundaryLoss, self).__init__()

    def forward(self, inputs, targets):
        sobel_x = torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=inputs.dtype, device=inputs.device).unsqueeze(0).unsqueeze(0) / 8.0
        sobel_y = torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=inputs.dtype, device=inputs.device).unsqueeze(0).unsqueeze(0) / 8.0
        edge_pred = F.conv2d(inputs, sobel_x, padding=1) + F.conv2d(inputs, sobel_y, padding=1)
        edge_true = F.conv2d(targets, sobel_x, padding=1) + F.conv2d(targets, sobel_y, padding=1)
        return F.l1_loss(edge_pred, edge_true)


class TotalLoss(nn.Module):
    def __init__(self, bce_weight=0.4, dice_weight=0.4, boundary_weight=0.2):
        super(TotalLoss, self).__init__()
        self.combo = ComboLoss(bce_weight, dice_weight)
        self.boundary = BoundaryLoss()
        self.boundary_weight = boundary_weight

    def forward(self, inputs, targets):
        return self.combo(inputs, targets) + self.boundary_weight * self.boundary(inputs, targets)


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
        Setup the U-Net or Attention U-Net model.
        Args:
            input_shape: Input shape for the model
        Returns:
            Model instance
        """
        logger.info("Setting up model...")
        model_type = self.config.get("model_type", "unet")
        if model_type == "attention_unet":
            logger.info("Using Attention U-Net model.")
            self.model = create_attention_unet_model(input_shape=input_shape)
        else:
            logger.info("Using classic U-Net model.")
            self.model = create_unet_model(input_shape=input_shape)
        self.model.to(self.device)
        
        # Setup optimizer
        if self.config["optimizer"] == "adam":
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.config["learning_rate"])
        elif self.config["optimizer"] == "adamw":
            self.optimizer = optim.AdamW(self.model.parameters(), lr=self.config["learning_rate"])
        elif self.config["optimizer"] == "sgd":
            self.optimizer = optim.SGD(self.model.parameters(), lr=self.config["learning_rate"])
        else:
            raise ValueError(f"Unsupported optimizer: {self.config['optimizer']}")
        
        # Setup loss function
        if self.config["loss_function"] == "bce":
            self.criterion = nn.BCELoss()
        elif self.config["loss_function"] == "focal":
            self.criterion = FocalLoss(alpha=0.8, gamma=2)
        elif self.config["loss_function"] == "combo":
            self.criterion = ComboLoss(bce_weight=self.config.get("bce_weight", 0.4),
                                       dice_weight=self.config.get("dice_weight", 0.4))
        elif self.config["loss_function"] == "total":
            self.criterion = TotalLoss(bce_weight=self.config.get("bce_weight", 0.4),
                                       dice_weight=self.config.get("dice_weight", 0.4),
                                       boundary_weight=self.config.get("boundary_weight", 0.2))
        else:
            raise ValueError(f"Unsupported loss function: {self.config['loss_function']}")
        
        logger.info("Model setup complete")
        self.model.summary()
        
        return self.model
    
    def setup_data(self, lazy_loading: bool = True) -> Tuple[DataLoader, DataLoader]:
        """
        Setup training and validation data.
        
        Args:
            lazy_loading: Whether to use lazy loading (recommended for large datasets)
            
        Returns:
            Tuple of (train_loader, val_loader)
        """
        logger.info("Setting up data...")
        
        self.data_loader = create_data_loader()
        
        if lazy_loading:
            logger.info("Using lazy loading...")
            train_dataset, val_dataset = self.data_loader.create_datasets(
                validation_split=self.config["validation_split"],
                random_seed=self.config["random_seed"]
            )
            
            # Log dataset info for debugging
            data_info = self.data_loader.get_data_info()
            logger.info(f"Dataset info: {data_info}")
            
            # Create data loaders
            train_loader = DataLoader(
                train_dataset, 
                batch_size=self.config["batch_size"], 
                shuffle=True,
                num_workers=0
            )
            val_loader = DataLoader(
                val_dataset, 
                batch_size=self.config["batch_size"], 
                shuffle=False,
                num_workers=0
            )
            
            return train_loader, val_loader
        
        else:
            # Traditional loading (loads all data into memory)
            logger.info("Using traditional data loading...")
            # DataLoader handles all logic: check timestamps, load existing or process raw
            data = self.data_loader.load_processed_data()
            
            # Log dataset info for debugging
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
    
    def _train_epoch(self, train_loader: DataLoader, epoch: int = 0, total_epochs: int = 1) -> Tuple[float, float, float, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        total_accuracy = 0.0
        total_dice = 0.0
        total_mse = 0.0
        num_batches = 0
        
        for batch_idx, (images, masks) in enumerate(tqdm(train_loader, desc=f"Training Epoch {epoch+1}/{total_epochs}")):
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
    
    def _validate_epoch(self, val_loader: DataLoader, epoch: int = 0, total_epochs: int = 1) -> Tuple[float, float, float, float]:
        """Validate for one epoch."""
        self.model.eval()
        total_loss = 0.0
        total_accuracy = 0.0
        total_dice = 0.0
        total_mse = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch_idx, (images, masks) in enumerate(tqdm(val_loader, desc=f"Validation Epoch {epoch+1}/{total_epochs}")):
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
        logger.info(f"Dataset Info:")
        logger.info(f" - Training samples: {len(train_loader.dataset)}")
        logger.info(f" - Validation samples: {len(val_loader.dataset)}")
        logger.info(f" - Batches per epoch: {len(train_loader)}")
        logger.info(f" - Total epochs: {epochs}")
        logger.info(f" - Batch size: {self.config['batch_size']}")
        logger.info(f" - Learning rate: {self.config['learning_rate']}")
        logger.info(f" - Optimizer: {self.config['optimizer']}")
        logger.info(f" - Model: {self.config['model_type']}")
        logger.info(f" - Loss function: {self.config['loss_function']}")
        logger.info(f" - Device: {self.device}")
        
        # Setup callbacks
        early_stopping = EarlyStopping(
            patience=CALLBACKS_CONFIG["early_stopping_patience"],
            monitor=CALLBACKS_CONFIG["early_stopping_monitor"]
        )
        
        best_val_loss = float('inf')
        
        for epoch in range(epochs):
            logger.info(f"\n{'='*60}")
            logger.info(f" Epoch {epoch+1}/{epochs} Started")
            logger.info(f"{'='*60}")
            
            # Training
            train_loss, train_acc, train_dice, train_mse = self._train_epoch(train_loader, epoch, epochs)
            
            # Validation
            val_loss, val_acc, val_dice, val_mse = self._validate_epoch(val_loader, epoch, epochs)
            
            # Log results with better formatting
            logger.info(f"Epoch {epoch+1}/{epochs} Results:")
            logger.info(f"   Train → Loss: {train_loss:.4f} | Acc: {train_acc:.4f} | Dice: {train_dice:.4f} | MSE: {train_mse:.6f}")
            logger.info(f"   Val   → Loss: {val_loss:.4f} | Acc: {val_acc:.4f} | Dice: {val_dice:.4f} | MSE: {val_mse:.6f}")
            
            # Epoch completed message
            logger.info(f" Epoch {epoch+1}/{epochs} Completed!")
            logger.info(f"{'='*60}")
            
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
            monitor_metric = val_loss if CALLBACKS_CONFIG["early_stopping_monitor"] == "val_loss" else val_dice
            if early_stopping(monitor_metric):
                logger.info(f"Early stopping triggered based on {CALLBACKS_CONFIG['early_stopping_monitor']}")
                break
            
            # Progress information
            remaining_epochs = epochs - (epoch + 1)
            progress_percent = ((epoch + 1) / epochs) * 100
            logger.info(f"Training Progress: {progress_percent:.1f}% | Remaining Epochs: {remaining_epochs}")
        
        logger.info(f"\n{'='*60}")
        logger.info("Training Completed!")
        logger.info(f"{'='*60}")
        logger.info(f"Final Results:")
        logger.info(f"   - Total Epochs: {len(self.history['train_loss'])}")
        logger.info(f"   - Best Val Loss: {min(self.history['val_loss']):.4f}")
        logger.info(f"   - Best Val Acc: {max(self.history['val_accuracy']):.3f}")
        logger.info(f"   - Best Val Dice: {max(self.history['val_dice']):.3f}")
        logger.info(f"{'='*60}")
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
    
    def load_trained_model(self, model_path: Path):
        """
        Load a trained model and its config.
        
        Args:
            model_path: Path to the model file (required)
            
        Returns:
            (Loaded model, config dict or None)
        """
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        config_path = model_path.parent / "config.json"
        config_data = None
        if config_path.exists():
            try:
                with open(config_path, 'r') as f:
                    config_data = json.load(f)
                model_config = config_data.get("model_config", {})
                model_type = model_config.get("model_type", "unet")
                logger.info(f"Loading model with config from: {config_path}")
                logger.info(f"Model config: {model_config}")
                
                model_config_clean = {k: v for k, v in model_config.items() if k != "model_type"}
                
                if model_type == "attention_unet":
                    self.model = create_attention_unet_model(**model_config_clean)
                else:
                    self.model = create_unet_model(**model_config_clean)
                self.model.to(self.device)
            except Exception as e:
                logger.warning(f"Failed to load config from {config_path}: {e}")
                logger.info("Falling back to default config")
                if self.model is None:
                    self.setup_model()
        else:
            logger.info(f"No config.json found at {config_path}, using default config")
            if self.model is None:
                self.setup_model()
        
        # Load model weights
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        
        logger.info(f"Loaded model from {model_path}")
        return self.model, config_data

    def save_trained_model(self, dataset_info: Dict[str, Any] = None) -> Path:
        from config import MODEL_CONFIG
        summary = self.get_training_summary()
        if not summary:
            raise ValueError("No training history available. Train the model first.")
        # Save model_type in config
        model_config = dict(MODEL_CONFIG)
        model_config["model_type"] = self.config.get("model_type", "unet")
        folder_path = create_timestamped_folder(TRAINED_MODELS_DIR, summary['best_val_accuracy'])
        save_config_json(folder_path, model_config, self.config, dataset_info, summary)
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