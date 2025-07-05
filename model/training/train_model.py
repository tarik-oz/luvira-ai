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

from training.trainer import create_trainer
from data.data_loader import create_data_loader
from config import TRAINING_CONFIG
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
    data_loader = None
    dataset_info = None
    try:
        logger.info("Starting hair segmentation training pipeline...")
        
        # Create trainer and data loader
        trainer = create_trainer()
        data_loader = create_data_loader()
        
        # Setup model
        logger.info("Setting up U-Net model...")
        trainer.setup_model()
        
        # Setup data
        logger.info("Setting up training data...")
        train_loader, val_loader = trainer.setup_data(load_processed=True)
        
        # Get dataset info for config
        dataset_info = data_loader.get_data_info()
        logger.info(f"Dataset info: {dataset_info}")
        
        # Start training
        logger.info("Starting model training...")
        history = trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=TRAINING_CONFIG["epochs"],
            batch_size=TRAINING_CONFIG["batch_size"]
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
            if trainer is not None and dataset_info is not None:
                logger.info("Saving trained model with metadata (finalize)...")
                model_folder = trainer.save_trained_model(dataset_info)
                logger.info(f"Model saved to: {model_folder}")
            else:
                logger.warning("Trainer or dataset_info not initialized, skipping model archiving.")
        except Exception as e:
            logger.error(f"Model archiving failed: {e}")


if __name__ == "__main__":
    main() 