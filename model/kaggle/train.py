#!/usr/bin/env python3
"""
Kaggle training script for hair segmentation U-Net model.
This script handles the complete training pipeline on Kaggle.

Usage:
    python train_kaggle.py

Make sure your dataset is named 'hairdataset' in Kaggle and contains 'images' and 'masks' folders.
"""

import sys
import os
import logging
import traceback
from pathlib import Path

# Add current directory to path for imports
sys.path.append(str(Path(__file__).parent))

# Configure logging for Kaggle
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('/kaggle/working/training.log')
    ]
)
logger = logging.getLogger(__name__)

def main():
    """
    Main training function for Kaggle.
    """
    trainer = None
    
    try:
        logger.info("=" * 60)
        logger.info("Starting Hair Segmentation Training on Kaggle")
        logger.info("=" * 60)
        
        # Import Kaggle-specific modules
        from .trainer import create_kaggle_trainer
        from .config import log_dataset_stats
        
        # Log dataset statistics
        logger.info("Checking dataset...")
        image_count, mask_count = log_dataset_stats()
        
        if image_count == 0 or mask_count == 0:
            raise RuntimeError("No images or masks found. Please check your dataset.")
        
        # Create Kaggle trainer
        logger.info("Creating Kaggle trainer...")
        trainer = create_kaggle_trainer()
        
        # Setup model and data
        logger.info("Setting up U-Net model...")
        trainer.setup_model()
        
        logger.info("Setting up training data...")
        train_loader, val_loader = trainer.setup_data()
        
        # Start training
        logger.info("Starting model training...")
        logger.info("Training will use lazy loading for memory efficiency")
        
        history = trainer.train(
            train_loader=train_loader,
            val_loader=val_loader
        )
        
        # Get training summary
        summary = trainer.get_training_summary()
        logger.info("Training completed successfully!")
        logger.info("=" * 60)
        logger.info("TRAINING SUMMARY")
        logger.info("=" * 60)
        for key, value in summary.items():
            logger.info(f"{key}: {value}")
        logger.info("=" * 60)
        
        # Save trained model
        logger.info("Saving trained model with metadata...")
        dataset_info = {
            "total_images": image_count,
            "total_masks": mask_count,
            "training_samples": len(train_loader.dataset),
            "validation_samples": len(val_loader.dataset)
        }
        
        if trainer.data_loader:
            dataset_info.update(trainer.data_loader.get_data_info())
        
        model_folder = trainer.save_trained_model(dataset_info)
        logger.info(f"Model saved to: {model_folder}")
        
        # Print final results
        logger.info("\n" + "=" * 60)
        logger.info("FINAL RESULTS")
        logger.info("=" * 60)
        logger.info(f"Best Validation Loss: {summary['best_val_loss']:.4f}")
        logger.info(f"Best Validation Dice: {summary['best_val_dice']:.4f}")
        logger.info(f"Training Duration: {summary['training_duration_minutes']:.2f} minutes")
        logger.info(f"Model Path: {summary['model_path']}")
        logger.info("=" * 60)
        
        # Plot training history if matplotlib is available
        try:
            plot_training_history(history)
        except ImportError:
            logger.warning("Matplotlib not available. Skipping plot generation.")
        except Exception as e:
            logger.warning(f"Could not generate plots: {e}")
        
        logger.info("Training pipeline completed successfully!")
        
    except Exception as e:
        logger.error(f"Training failed with error: {e}")
        logger.error(f"Error details: {traceback.format_exc()}")
        raise
    
    finally:
        # Final cleanup and model save attempt
        if trainer is not None:
            try:
                logger.info("Performing final model save...")
                dataset_info = {"emergency_save": True}
                if trainer.data_loader:
                    dataset_info.update(trainer.data_loader.get_data_info())
                
                model_folder = trainer.save_trained_model(dataset_info)
                logger.info(f"Emergency model save completed: {model_folder}")
                
            except Exception as e:
                logger.error(f"Emergency model save failed: {e}")


def plot_training_history(history):
    """
    Plot training history if matplotlib is available.
    
    Args:
        history: Training history dictionary
    """
    try:
        import matplotlib.pyplot as plt
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Training History', fontsize=16)
        
        # Plot loss
        axes[0, 0].plot(history['train_loss'], label='Train Loss')
        axes[0, 0].plot(history['val_loss'], label='Val Loss')
        axes[0, 0].set_title('Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Plot accuracy
        axes[0, 1].plot(history['train_accuracy'], label='Train Acc')
        axes[0, 1].plot(history['val_accuracy'], label='Val Acc')
        axes[0, 1].set_title('Accuracy')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Plot Dice coefficient
        axes[1, 0].plot(history['train_dice'], label='Train Dice')
        axes[1, 0].plot(history['val_dice'], label='Val Dice')
        axes[1, 0].set_title('Dice Coefficient')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Dice')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # Plot MSE
        axes[1, 1].plot(history['train_mse'], label='Train MSE')
        axes[1, 1].plot(history['val_mse'], label='Val MSE')
        axes[1, 1].set_title('Mean Squared Error')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('MSE')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig('/kaggle/working/training_history.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        logger.info("Training history plot saved to /kaggle/working/training_history.png")
        
    except Exception as e:
        logger.warning(f"Could not create training plots: {e}")


def check_environment():
    """
    Check Kaggle environment and dependencies.
    """
    logger.info("Checking Kaggle environment...")
    
    # Check if running on Kaggle
    if not Path("/kaggle").exists():
        logger.warning("Not running on Kaggle. This script is optimized for Kaggle.")
    
    # Check GPU availability
    try:
        import torch
        logger.info(f"PyTorch version: {torch.__version__}")
        logger.info(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            logger.info(f"CUDA version: {torch.version.cuda}")
            logger.info(f"GPU device: {torch.cuda.get_device_name(0)}")
    except ImportError:
        logger.error("PyTorch not available!")
        
    # Check required directories
    required_dirs = ["/kaggle/input", "/kaggle/working"]
    for dir_path in required_dirs:
        if Path(dir_path).exists():
            logger.info(f"✓ {dir_path} exists")
        else:
            logger.error(f"✗ {dir_path} not found")
    
    logger.info("Environment check completed.")


if __name__ == "__main__":
    # Check environment first
    check_environment()
    
    # Run main training
    main() 