"""
Training module for hair segmentation U-Net model.
Handles model training with callbacks and logging.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
import logging
import json
from tqdm import tqdm
import numpy as np

from model.config import (
    TRAINING_CONFIG, CALLBACKS_CONFIG, 
    TRAINED_MODELS_DIR, MODEL_CONFIG,
    DATA_CONFIG
)
from model.models.unet_model import create_unet_model
from model.models.attention_unet_model import create_attention_unet_model
from model.data_loader.factory_data_loader import create_auto_data_loader
from model.utils.model_saving import (
    create_timestamped_folder, save_config_json, 
    save_training_log, save_models_to_folder
)
from model.training.callbacks import EarlyStopping, FocalLoss, DiceLoss, ComboLoss, BoundaryLoss, TotalLoss
from model.training.metrics import calculate_accuracy, calculate_dice, calculate_mse

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
        self.model_path = TRAINED_MODELS_DIR / "best_model.pth"
        self.latest_model_path = TRAINED_MODELS_DIR / "latest_model.pth"
        
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
        self.scheduler = None
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
        if self.config["device"] in ["auto", "cuda"]:
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
        model_type = MODEL_CONFIG.get("model_type", "unet")
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
        
        # Setup learning rate scheduler
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=CALLBACKS_CONFIG.get("reduce_lr_factor", 0.1),
            patience=CALLBACKS_CONFIG.get("reduce_lr_patience", 3),
            min_lr=CALLBACKS_CONFIG.get("reduce_lr_min_lr", 1e-6),
            verbose=True
        )
        
        logger.info("Model setup complete")
        
        # Print model summary if available
        if hasattr(self.model, 'summary'):
            self.model.summary()
        
        return self.model
    
    def setup_data(self, lazy_loading: bool = None) -> Tuple[DataLoader, DataLoader]:
        """
        Setup training and validation data using factory pattern.
        
        Args:
            lazy_loading: Whether to use lazy loading (if None, uses config value)
            
        Returns:
            Tuple of (train_loader, val_loader)
        """
        # Use config value if not specified
        if lazy_loading is None:
            lazy_loading = DATA_CONFIG.get("lazy_loading", False)
        
        logger.info(f"Setting up data with {'lazy' if lazy_loading else 'traditional'} loading...")
        
        # Create appropriate data loader using factory
        self.data_loader = create_auto_data_loader(lazy_loading=lazy_loading)
        
        # Get datasets from data loader
        train_dataset, val_dataset = self.data_loader.get_datasets(
            validation_split=self.config["validation_split"],
            random_seed=self.config["random_seed"]
        )
        
        # Log dataset info
        data_info = self.data_loader.get_data_info()
        logger.info(f"Dataset info: {data_info}")
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset, 
            batch_size=self.config["batch_size"], 
            shuffle=True,
            num_workers=DATA_CONFIG.get("num_workers", 0),
            pin_memory=torch.cuda.is_available(),  # Auto-detect GPU
            drop_last=True
        )
        val_loader = DataLoader(
            val_dataset, 
            batch_size=self.config["batch_size"], 
            shuffle=False,
            num_workers=DATA_CONFIG.get("num_workers", 0),
            pin_memory=torch.cuda.is_available(),  # Auto-detect GPU
            drop_last=False 
        )
        
        logger.info(f"Created data loaders: {len(train_loader)} train batches, {len(val_loader)} val batches")
        return train_loader, val_loader
    
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
            
            # Calculate metrics
            accuracy = calculate_accuracy(predictions, masks)
            dice = calculate_dice(predictions, masks)
            mse = calculate_mse(predictions, masks)
            
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
                
                # Calculate metrics
                accuracy = calculate_accuracy(predictions, masks)
                dice = calculate_dice(predictions, masks)
                mse = calculate_mse(predictions, masks)
                
                total_loss += loss.item()
                total_accuracy += accuracy
                total_dice += dice
                total_mse += mse
                num_batches += 1
        
        return (total_loss / num_batches, total_accuracy / num_batches, total_dice / num_batches, total_mse / num_batches)
    
    def _save_best_model(self, val_loss: float, val_dice: float, best_val_loss: float, best_val_dice: float) -> Tuple[float, float]:
        """Save the best model based on configured monitor metric."""
        checkpoint_monitor = CALLBACKS_CONFIG.get("checkpoint_monitor", "val_loss")
        if checkpoint_monitor == "val_dice":
            if val_dice > best_val_dice:
                best_val_dice = val_dice
                torch.save(self.model.state_dict(), self.model_path)
                logger.info(f"Saved best model with val_dice: {val_dice:.4f}")
        else:  # val_loss
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(self.model.state_dict(), self.model_path)
                logger.info(f"Saved best model with val_loss: {val_loss:.4f}")
        
        # Save latest model
        torch.save(self.model.state_dict(), self.latest_model_path)
        
        return best_val_loss, best_val_dice
    
    def _update_learning_rate(self, val_loss: float, val_dice: float):
        """Update learning rate using ReduceLROnPlateau scheduler."""
        current_lr = self.optimizer.param_groups[0]['lr']
        reduce_lr_metric = val_loss if CALLBACKS_CONFIG.get("reduce_lr_monitor", "val_loss") == "val_loss" else val_dice
        self.scheduler.step(reduce_lr_metric)
        new_lr = self.optimizer.param_groups[0]['lr']
        if new_lr < current_lr:
            logger.info(f"Learning rate reduced from {current_lr:.2e} to {new_lr:.2e}")

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
        
        epochs = self.config['epochs']
        
        logger.info("Starting training...")
        logger.info(f"Dataset Info:")
        logger.info(f" - Training samples: {len(train_loader.dataset)}")
        logger.info(f" - Validation samples: {len(val_loader.dataset)}")
        logger.info(f" - Batches per epoch: {len(train_loader)}")
        logger.info(f" - Total epochs: {epochs}")
        logger.info(f" - Batch size: {self.config['batch_size']}")
        logger.info(f" - Learning rate: {self.config['learning_rate']}")
        logger.info(f" - Optimizer: {self.config['optimizer']}")
        logger.info(f" - Model: {MODEL_CONFIG['model_type']}")
        logger.info(f" - Loss function: {self.config['loss_function']}")
        logger.info(f" - Device: {self.device}")
        
        # Setup callbacks
        early_stopping = EarlyStopping(
            patience=CALLBACKS_CONFIG["early_stopping_patience"],
            monitor=CALLBACKS_CONFIG["early_stopping_monitor"]
        )
        
        best_val_loss = float('inf')
        best_val_dice = 0.0
        
        for epoch in range(epochs):
            logger.info(f"\n{'='*60}")
            logger.info(f" Epoch {epoch+1}/{epochs}")
            logger.info(f"{'='*60}")
            
            # Training
            train_loss, train_acc, train_dice, train_mse = self._train_epoch(train_loader, epoch, epochs)
            
            # Validation
            val_loss, val_acc, val_dice, val_mse = self._validate_epoch(val_loader, epoch, epochs)
            
            # Log results with better formatting
            logger.info(f"Epoch {epoch+1}/{epochs} Results:")
            logger.info(f"   Train → Loss: {train_loss:.4f} | Acc: {train_acc:.4f} | Dice: {train_dice:.4f} | MSE: {train_mse:.6f}")
            logger.info(f"   Val   → Loss: {val_loss:.4f} | Acc: {val_acc:.4f} | Dice: {val_dice:.4f} | MSE: {val_mse:.6f}")
            
            # Save history
            self.history['train_loss'].append(train_loss)
            self.history['train_accuracy'].append(train_acc)
            self.history['train_dice'].append(train_dice)
            self.history['train_mse'].append(train_mse)
            self.history['val_loss'].append(val_loss)
            self.history['val_accuracy'].append(val_acc)
            self.history['val_dice'].append(val_dice)
            self.history['val_mse'].append(val_mse)
            
            # Save best model (with configurable monitor)
            best_val_loss, best_val_dice = self._save_best_model(val_loss, val_dice, best_val_loss, best_val_dice)
            
            # Learning rate scheduling (reduce on plateau)
            self._update_learning_rate(val_loss, val_dice)
            
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
        
        # Find the best epoch based on the checkpoint monitor
        monitor = CALLBACKS_CONFIG.get("checkpoint_monitor", "val_loss")
        if "loss" in monitor or "mse" in monitor:
            best_epoch_idx = np.argmin(self.history[monitor])
        else: # dice, accuracy
            best_epoch_idx = np.argmax(self.history[monitor])

        return {
            'final_train_loss': self.history['train_loss'][-1],
            'final_train_accuracy': self.history['train_accuracy'][-1],
            'final_train_dice': self.history['train_dice'][-1],
            'final_train_mse': self.history['train_mse'][-1],
            'final_val_loss': self.history['val_loss'][-1],
            'final_val_accuracy': self.history['val_accuracy'][-1],
            'final_val_dice': self.history['val_dice'][-1],
            'final_val_mse': self.history['val_mse'][-1],
            'best_val_loss': self.history['val_loss'][best_epoch_idx],
            'best_val_accuracy': self.history['val_accuracy'][best_epoch_idx],
            'best_val_dice': self.history['val_dice'][best_epoch_idx],
            'best_val_mse': self.history['val_mse'][best_epoch_idx],
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
                logger.info(f"Loading model configuration from: {config_path}")
                
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
        summary = self.get_training_summary()
        if not summary:
            raise ValueError("No training history available. Train the model first.")
        # Save model_type in config
        model_config = dict(MODEL_CONFIG)
        training_config = dict(self.config)

        folder_path = create_timestamped_folder(TRAINED_MODELS_DIR, summary['best_val_accuracy'])
        save_config_json(folder_path, model_config, training_config, CALLBACKS_CONFIG, dataset_info, summary)
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