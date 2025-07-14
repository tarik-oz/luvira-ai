"""
Kaggle-specific training module for hair segmentation U-Net model.
Uses Kaggle configuration and handles Kaggle-specific paths.
UPDATED VERSION - All fixes included!
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
import logging
from tqdm import tqdm
import time
import signal
import sys

# Kaggle config imports
try:
    from .config import (
        TRAINING_CONFIG, CALLBACKS_CONFIG, 
        TRAINED_MODELS_DIR, MODEL_CONFIG,
        DATA_CONFIG, KAGGLE_WORKING_DIR
    )
except ImportError:
    from config import (
        TRAINING_CONFIG, CALLBACKS_CONFIG, 
        TRAINED_MODELS_DIR, MODEL_CONFIG,
        DATA_CONFIG, KAGGLE_WORKING_DIR
    )

# Model imports
try:
    from ..models.unet_model import create_unet_model
    from ..models.attention_unet_model import create_attention_unet_model
except ImportError:
    from models.unet_model import create_unet_model
    from models.attention_unet_model import create_attention_unet_model

# Data loader imports
try:
    from .factory import create_auto_data_loader
except ImportError:
    from factory import create_auto_data_loader

# Utils imports
try:
    from ..utils.model_saving import (
        create_timestamped_folder, save_config_json, 
        save_training_log, save_models_to_folder
    )
except ImportError:
    from utils.model_saving import (
        create_timestamped_folder, save_config_json, 
        save_training_log, save_models_to_folder
    )

# Training imports
try:
    from .callbacks import EarlyStopping, FocalLoss, DiceLoss, ComboLoss, BoundaryLoss, TotalLoss
    from .metrics import calculate_accuracy, calculate_dice, calculate_mse
except ImportError:
    try:
        from model.kaggle.callbacks import EarlyStopping, FocalLoss, DiceLoss, ComboLoss, BoundaryLoss, TotalLoss
        from model.kaggle.metrics import calculate_accuracy, calculate_dice, calculate_mse
    except ImportError:
        from callbacks import EarlyStopping, FocalLoss, DiceLoss, ComboLoss, BoundaryLoss, TotalLoss
        from metrics import calculate_accuracy, calculate_dice, calculate_mse

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class KaggleHairSegmentationTrainer:
    """
    Kaggle-specific trainer class for hair segmentation U-Net model.
    
    Handles model training, callbacks, and logging in Kaggle environment.
    UPDATED VERSION - All fixes included!
    """
    
    def __init__(self, **training_config):
        """
        Initialize the Kaggle trainer.
        
        Args:
            **training_config: Training configuration parameters
        """
        # Create model paths for Kaggle
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
        
        # Training metrics tracking
        self.best_val_loss = float('inf')
        self.best_val_dice = float('-inf')
        
        # Training timing
        self.training_start_time = None
        self.training_end_time = None
        
        # Setup device
        self._setup_device()
        
        # Print Kaggle environment info
        self._print_kaggle_info()
        
        # Register signal handler for interruptions
        self._register_signal_handler()
        
    def _setup_device(self):
        """Setup device (CPU/GPU) for training in Kaggle."""
        if self.config["device"] in ["auto", "cuda"]:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device("cpu")
            
        logger.info(f"Using device: {self.device}")
        
        # Print GPU info if available
        if torch.cuda.is_available():
            logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
            logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        
    def _print_kaggle_info(self):
        """Print Kaggle environment information."""
        logger.info("=== Kaggle Environment Info ===")
        logger.info(f"Working directory: {KAGGLE_WORKING_DIR}")
        logger.info(f"Model save directory: {TRAINED_MODELS_DIR}")
        logger.info(f"Lazy loading: {DATA_CONFIG['lazy_loading']}")
        logger.info(f"Batch size: {self.config['batch_size']}")
        logger.info(f"Epochs: {self.config['epochs']}")
        logger.info(f"Learning rate: {self.config['learning_rate']}")
        logger.info("===============================")
        
    def setup_model(self, input_shape: tuple = None) -> nn.Module:
        """
        Setup the U-Net or Attention U-Net model for Kaggle.
        
        Args:
            input_shape: Input shape for the model
            
        Returns:
            Model instance
        """
        logger.info("Setting up model for Kaggle...")
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
        
        # Setup learning rate scheduler
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=CALLBACKS_CONFIG.get("reduce_lr_factor", 0.2),
            patience=CALLBACKS_CONFIG.get("reduce_lr_patience", 12),
            min_lr=CALLBACKS_CONFIG.get("reduce_lr_min_lr", 1e-7),
            verbose=True
        )
        
        logger.info("Model setup complete for Kaggle")
        return self.model
    
    def setup_data(self, lazy_loading: bool = None) -> Tuple[DataLoader, DataLoader]:
        """
        Setup training and validation data for Kaggle.
        
        Args:
            lazy_loading: Whether to use lazy loading (if None, uses config value)
            
        Returns:
            Tuple of (train_loader, val_loader)
        """
        # Use config value if not specified
        if lazy_loading is None:
            lazy_loading = DATA_CONFIG.get("lazy_loading", True)  # Default to True for Kaggle
        
        logger.info(f"Setting up data for Kaggle with {'lazy' if lazy_loading else 'traditional'} loading...")
        
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
        
        # Create data loaders optimized for Kaggle
        train_loader = DataLoader(
            train_dataset, 
            batch_size=self.config["batch_size"], 
            shuffle=True,
            num_workers=DATA_CONFIG.get("num_workers", 2),
            pin_memory=torch.cuda.is_available(),
            drop_last=True
        )
        val_loader = DataLoader(
            val_dataset, 
            batch_size=self.config["batch_size"], 
            shuffle=False,
            num_workers=DATA_CONFIG.get("num_workers", 2),
            pin_memory=torch.cuda.is_available(),
            drop_last=False 
        )
        
        logger.info(f"Created data loaders: {len(train_loader)} train batches, {len(val_loader)} val batches")
        logger.info(f"Training samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}")
        
        return train_loader, val_loader
    
    def _train_epoch(self, train_loader: DataLoader, epoch: int = 0, total_epochs: int = 1) -> Tuple[float, float, float, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        total_accuracy = 0.0
        total_dice = 0.0
        total_mse = 0.0
        num_batches = 0
        
        progress_bar = tqdm(train_loader, desc=f"Training Epoch {epoch+1}/{total_epochs}")
        
        for batch_idx, (images, masks) in enumerate(progress_bar):
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
            
            # Update progress bar
            progress_bar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{accuracy:.4f}',
                'Dice': f'{dice:.4f}'
            })
        
        avg_loss = total_loss / num_batches
        avg_accuracy = total_accuracy / num_batches
        avg_dice = total_dice / num_batches
        avg_mse = total_mse / num_batches
        
        return avg_loss, avg_accuracy, avg_dice, avg_mse
    
    def _validate_epoch(self, val_loader: DataLoader, epoch: int = 0, total_epochs: int = 1) -> Tuple[float, float, float, float]:
        """Validate for one epoch."""
        self.model.eval()
        total_loss = 0.0
        total_accuracy = 0.0
        total_dice = 0.0
        total_mse = 0.0
        num_batches = 0
        
        with torch.no_grad():
            progress_bar = tqdm(val_loader, desc=f"Validation Epoch {epoch+1}/{total_epochs}")
            
            for batch_idx, (images, masks) in enumerate(progress_bar):
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
                
                # Update progress bar
                progress_bar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Acc': f'{accuracy:.4f}',
                    'Dice': f'{dice:.4f}'
                })
        
        avg_loss = total_loss / num_batches
        avg_accuracy = total_accuracy / num_batches
        avg_dice = total_dice / num_batches
        avg_mse = total_mse / num_batches
        
        return avg_loss, avg_accuracy, avg_dice, avg_mse
    
    def _save_best_model(self, val_loss: float, val_dice: float, best_val_loss: float, best_val_dice: float) -> Tuple[float, float]:
        """
        Save the model if it has the best validation metrics.
        
        Args:
            val_loss: Current validation loss
            val_dice: Current validation Dice score
            best_val_loss: Best validation loss so far
            best_val_dice: Best validation Dice score so far
            
        Returns:
            Tuple of (new best val loss, new best val dice)
        """
        # Update best metrics
        monitor = CALLBACKS_CONFIG.get("checkpoint_monitor", "val_loss")
        
        if monitor == "val_loss" and val_loss < best_val_loss:
            logger.info(f"Validation loss improved from {best_val_loss:.4f} to {val_loss:.4f}")
            best_val_loss = val_loss
            torch.save(self.model.state_dict(), self.model_path)
        elif monitor == "val_dice" and val_dice > best_val_dice:
            logger.info(f"Validation Dice improved from {best_val_dice:.4f} to {val_dice:.4f}")
            best_val_dice = val_dice
            torch.save(self.model.state_dict(), self.model_path)
        
        # Update instance variables to track best metrics
        self.best_val_loss = best_val_loss
        self.best_val_dice = best_val_dice
        
        # Always save the latest model
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
        Train the model for Kaggle.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            **kwargs: Additional training parameters
            
        Returns:
            Training history
        """
        logger.info("Starting training on Kaggle...")
        self.training_start_time = time.time()
        
        epochs = kwargs.get('epochs', self.config["epochs"])
        
        # Setup early stopping (FIXED: removed verbose parameter)
        early_stopping = EarlyStopping(
            patience=CALLBACKS_CONFIG["early_stopping_patience"],
            monitor=CALLBACKS_CONFIG["early_stopping_monitor"]
        )
        
        for epoch in range(epochs):
            logger.info(f"\nEpoch {epoch+1}/{epochs}")
            
            # Train
            train_loss, train_acc, train_dice, train_mse = self._train_epoch(
                train_loader, epoch, epochs
            )
            
            # Validate
            val_loss, val_acc, val_dice, val_mse = self._validate_epoch(
                val_loader, epoch, epochs
            )
            
            # Save metrics
            self.history['train_loss'].append(train_loss)
            self.history['train_accuracy'].append(train_acc)
            self.history['train_dice'].append(train_dice)
            self.history['train_mse'].append(train_mse)
            self.history['val_loss'].append(val_loss)
            self.history['val_accuracy'].append(val_acc)
            self.history['val_dice'].append(val_dice)
            self.history['val_mse'].append(val_mse)
            
            # Print epoch summary
            logger.info(f"Train - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}, Dice: {train_dice:.4f}")
            logger.info(f"Val   - Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, Dice: {val_dice:.4f}")
            
            # Save best model (with configurable monitor)
            self.best_val_loss, self.best_val_dice = self._save_best_model(val_loss, val_dice, self.best_val_loss, self.best_val_dice)
            
            # Learning rate scheduling (reduce on plateau)
            self._update_learning_rate(val_loss, val_dice)
            
            # Early stopping check
            if early_stopping(val_dice):
                logger.info(f"Early stopping triggered at epoch {epoch+1}")
                break
        
        self.training_end_time = time.time()
        training_duration = self.training_end_time - self.training_start_time
        
        logger.info(f"Training completed in {training_duration:.2f} seconds ({training_duration/60:.2f} minutes)")
        
        return self.history
    
    def get_training_summary(self) -> Dict[str, Any]:
        """Get training summary for Kaggle."""
        if not self.history['train_loss']:
            return {"status": "No training data available"}
        
        duration = 0
        if self.training_start_time and self.training_end_time:
            duration = self.training_end_time - self.training_start_time
        
        return {
            "status": "completed",
            "epochs_trained": len(self.history['train_loss']),
            "final_train_loss": self.history['train_loss'][-1],
            "final_val_loss": self.history['val_loss'][-1],
            "best_val_loss": min(self.history['val_loss']),
            "final_train_dice": self.history['train_dice'][-1],
            "final_val_dice": self.history['val_dice'][-1],
            "best_val_dice": max(self.history['val_dice']),
            "training_duration_seconds": duration,
            "training_duration_minutes": duration / 60,
            "model_path": str(self.model_path),
            "device": str(self.device)
        }
    
    def save_trained_model(self, dataset_info: Dict[str, Any] = None) -> Path:
        """
        Save the trained model with metadata for Kaggle.
        FIXED: Corrected function parameters and calls.
        
        Args:
            dataset_info: Dataset information
            
        Returns:
            Path to saved model folder
        """
        logger.info("Saving trained model for Kaggle...")
        
        # Get best validation accuracy/dice for folder naming
        best_val_acc = max(self.history['val_accuracy']) if self.history['val_accuracy'] else 0.0
        
        # FIXED: Create timestamped folder with correct parameters
        model_folder = create_timestamped_folder(TRAINED_MODELS_DIR, best_val_acc)
        
        # FIXED: Save model files with correct parameters
        save_models_to_folder(model_folder, self.model_path, self.latest_model_path)
        
        # Get training summary
        summary = self.get_training_summary()
        
        # FIXED: Save configuration with correct parameters
        save_config_json(model_folder, MODEL_CONFIG, self.config, dataset_info or {}, summary)
        
        # FIXED: Save training log with correct parameters
        save_training_log(model_folder, self.device, self.history, summary)
        
        logger.info(f"Model saved to: {model_folder}")
        return model_folder

    def _register_signal_handler(self):
        """Register signal handlers to save model on interruption."""
        def signal_handler(sig, frame):
            print("\n\n⚠️ Training interrupted! Saving full model directory before exiting...")
            try:
                if self.model is not None:
                    # Use the regular save function to create a complete directory with logs and config
                    dataset_info = {
                        "interrupted": True,
                        "completed_epochs": len(self.history['train_loss']),
                        "best_val_loss": self.best_val_loss,
                        "best_val_dice": self.best_val_dice
                    }
                    
                    # If we have data loader information, add it
                    if self.data_loader:
                        dataset_info.update(self.data_loader.get_data_info())
                    
                    # Save complete model directory
                    model_dir = self.save_trained_model(dataset_info)
                    print(f"✅ Full model directory saved at: {model_dir}")
            except Exception as e:
                print(f"❌ Failed to save model directory on interrupt: {e}")
            
            print("Exiting...")
            sys.exit(0)
        
        # Register handlers for interrupt signals
        signal.signal(signal.SIGINT, signal_handler)  # Ctrl+C
        signal.signal(signal.SIGTERM, signal_handler)  # termination signal


def create_kaggle_trainer(**kwargs) -> KaggleHairSegmentationTrainer:
    """
    Create a Kaggle trainer instance.
    
    Args:
        **kwargs: Configuration parameters
        
    Returns:
        KaggleHairSegmentationTrainer instance
    """
    return KaggleHairSegmentationTrainer(**kwargs) 