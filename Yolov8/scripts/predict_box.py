"""
YOLOv8 Hair Detection (Bounding Box) Script

This script performs hair detection using YOLOv8 bounding box prediction.
"""

from ultralytics import YOLO
import logging
import os
from config import BEST_MODEL_PATH, TEST_IMAGES_DIR, TRAINING_CONFIG

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def predict_hair_boxes():
    """
    Perform hair detection using YOLOv8 bounding box prediction.
    
    This function loads a trained YOLOv8 model and performs detection
    on test images, showing bounding boxes around detected hair regions.
    """
    try:
        # Model path from config
        model_path = str(BEST_MODEL_PATH)
        
        # Check if model exists
        if not os.path.exists(model_path):
            logger.error(f"Model not found: {model_path}")
            logger.info("Please train the model first using train.py")
            return
        
        logger.info(f"Loading model from {model_path}")
        model = YOLO(model_path)
        
        # Test image path from config
        test_image = str(TEST_IMAGES_DIR / "1624.jpg")
        
        # Check if test image exists
        if not os.path.exists(test_image):
            logger.error(f"Test image not found: {test_image}")
            logger.info(f"Please ensure test image exists in {TEST_IMAGES_DIR}")
            return
        
        logger.info(f"Running prediction on {test_image}")
        
        # Run prediction with bounding boxes using config values
        results = model.predict(
            source=test_image, 
            show=True, 
            save=True, 
            conf=TRAINING_CONFIG["confidence"],  # Confidence threshold from config
            iou=TRAINING_CONFIG["iou_threshold"]  # IoU threshold from config
        )
        
        logger.info("Prediction completed successfully!")
        logger.info("Results saved in 'runs/detect/predict' directory")
        
        return results
        
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise

if __name__ == "__main__":
    predict_hair_boxes()
