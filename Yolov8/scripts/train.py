"""
YOLOv8 Hair Segmentation Training Script

This script trains a YOLOv8 segmentation model for hair detection.
"""

from ultralytics import YOLO
import logging
from config import TRAINING_CONFIG, save_yaml_config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def train_yolov8_model():
    """
    Train YOLOv8 segmentation model for hair detection.
    
    The model will be trained using the configuration in config.yaml
    and saved to model/best.pt
    """
    try:
        logger.info("Starting YOLOv8 hair segmentation training...")
        
        # Generate YAML config from Python config
        save_yaml_config()
        logger.info("Generated config.yaml from config.py")
        
        # Load base model
        model = YOLO("yolov8n-seg.pt")
        logger.info("Loaded YOLOv8n-seg base model")
        
        # Train the model
        logger.info("Starting training with config.yaml...")
        results = model.train(
            data="config.yaml", 
            epochs=TRAINING_CONFIG["epochs"], 
            batch=TRAINING_CONFIG["batch_size"],
            imgsz=TRAINING_CONFIG["image_size"],
            save=TRAINING_CONFIG["save"],
            project=TRAINING_CONFIG["project"]
        )
        
        logger.info("Training completed successfully!")
        logger.info(f"Best model saved to: {results.save_dir}")
        
        return results
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise

if __name__ == "__main__":
    train_yolov8_model()
