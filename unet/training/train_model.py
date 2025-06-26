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
    try:
        logger.info("Starting hair segmentation training pipeline...")
        
        # Create trainer
        trainer = create_trainer()
        
        # Setup model
        logger.info("Setting up U-Net model...")
        trainer.setup_model()
        
        # Setup data
        logger.info("Setting up training data...")
        train_loader, val_loader = trainer.setup_data(load_processed=True)
        
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


if __name__ == "__main__":
    main() 