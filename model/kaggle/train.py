#!/usr/bin/env python3
"""
Main training script for hair segmentation U-Net model.
This script handles the complete training pipeline.
"""

import sys
from pathlib import Path

# Add working directory to path for absolute imports
sys.path.insert(0, '/kaggle/working')

from training.trainer import create_trainer

def main():
    """
    Main training function.
    """
    trainer = None
    try:
        print("Starting hair segmentation training pipeline...")
        
        # Create trainer
        trainer = create_trainer()
        
        # Setup model and data (data_loader otomatik oluşturulur)
        print("Setting up model...")
        trainer.setup_model()
        
        print("Setting up training data...")
        train_loader, val_loader = trainer.setup_data()  # Uses config value for lazy loading
        
        # Start training
        history = trainer.train(
            train_loader=train_loader,
            val_loader=val_loader
        )
        
        # Get training summary
        summary = trainer.get_training_summary()
        print("Training completed successfully!")
        
    except Exception as e:
        print(f"Training failed: {e}")
        import traceback
        traceback.print_exc()
        raise
    finally:
        # Save the model regardless of whether training completed successfully or not
        try:
            if trainer is not None:
                print("Saving trained model with metadata (finalize)...")
                if trainer.data_loader:
                    dataset_info = trainer.data_loader.get_data_info()
                    model_folder = trainer.save_trained_model(dataset_info)
                    print(f"Model saved to: {model_folder}")
                else:
                    print("Data loader not initialized, skipping model save with metadata.")
            else:
                print("Trainer not initialized, skipping model archiving.")
        except Exception as e:
            print(f"Model archiving failed: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()