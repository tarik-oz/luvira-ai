#!/usr/bin/env python3
"""
Main training script for hair segmentation U-Net model.
This script handles the complete training pipeline.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

try:
    from .trainer import create_trainer
    from ..data_loader.factory_data_loader import create_auto_data_loader
except ImportError:
    from trainer import create_trainer
    from data_loader.factory_data_loader import create_auto_data_loader
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """
    Main training function.
    """
    trainer = None
    try:
        logger.info("Starting hair segmentation training pipeline...")
        
        # Create trainer
        trainer = create_trainer()
        
        # Setup model and data (data_loader otomatik oluşturulur)
        logger.info("Setting up U-Net model...")
        trainer.setup_model()
        
        logger.info("Setting up training data...")
        train_loader, val_loader = trainer.setup_data()  # Uses config value for lazy loading
        
        # Start training
        logger.info("Starting model training...")
        history = trainer.train(
            train_loader=train_loader,
            val_loader=val_loader
        )
        
        # Get training summary
        summary = trainer.get_training_summary()
        logger.info("Training completed successfully!")
        logger.info(f"Training summary: {summary}")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise
    finally:
        # Save the model regardless of whether training completed successfully or not
        try:
            if trainer is not None:
                logger.info("Saving trained model with metadata (finalize)...")
                dataset_info = trainer.data_loader.get_data_info()
                model_folder = trainer.save_trained_model(dataset_info)
                logger.info(f"Model saved to: {model_folder}")
            else:
                logger.warning("Trainer not initialized, skipping model archiving.")
        except Exception as e:
            logger.error(f"Model archiving failed: {e}")


if __name__ == "__main__":
    main() 