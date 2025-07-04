"""
YOLOv8 Hair Segmentation with Resize Prediction Script

This script performs hair segmentation using a trained YOLOv8 model with automatic resizing.
"""

from ultralytics import YOLO
import numpy as np
import cv2
import os
import logging
from pathlib import Path
from config import BEST_MODEL_PATH, TEST_IMAGES_DIR, OUTPUT_DIR

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class YOLOv8ResizePredictor:
    """YOLOv8-based hair segmentation predictor with resize functionality."""
    
    def __init__(self, model_path: str = None):
        """
        Initialize the predictor.
        
        Args:
            model_path: Path to the trained YOLOv8 model (uses config default if None)
        """
        self.model_path = model_path or str(BEST_MODEL_PATH)
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Load the YOLOv8 model."""
        try:
            logger.info(f"Loading model from {self.model_path}")
            self.model = YOLO(self.model_path)
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def predict_with_resize(self, image_path: str, output_dir: str = None, scale_percent: int = 100):
        """
        Predict hair mask for an image with resize functionality.
        
        Args:
            image_path: Path to input image
            output_dir: Directory to save results (uses config default if None)
            scale_percent: Scale percentage for display (100 = original size)
            
        Returns:
            Tuple of (final_mask, resized_image, resized_mask)
        """
        if output_dir is None:
            output_dir = str(OUTPUT_DIR)
            
        try:
            # Load image
            img = cv2.imread(image_path)
            if img is None:
                raise ValueError(f"Could not load image: {image_path}")
            
            img_height, img_width, _ = img.shape
            logger.info(f"Processing image: {image_path} ({img_width}x{img_height})")
            
            # Create output directory
            os.makedirs(output_dir, exist_ok=True)
            
            # Run prediction
            results = self.model(img)
            result = results[0]  # Get first result
            
            # Get model class names
            names = self.model.names
            logger.info(f"Model classes: {names}")
            
            # Initialize final mask
            final_mask = np.zeros((img_height, img_width), dtype=np.uint8)
            
            # Get predicted classes
            predicted_classes = result.boxes.cls.cpu().numpy()
            logger.info(f"Predicted classes: {predicted_classes}")
            
            # Process each mask
            for j, mask in enumerate(result.masks.data):
                # Convert mask to numpy and scale
                mask_np = mask.cpu().numpy() * 255
                class_id = int(predicted_classes[j])
                
                logger.info(f"Object {j} detected as {class_id} - {names[class_id]}")
                
                # Resize mask to original image size
                mask_resized = cv2.resize(mask_np, (img_width, img_height))
                
                # Combine with final mask (take maximum)
                final_mask = np.maximum(final_mask, mask_resized)
                
                # Save individual mask
                mask_filename = f"output{j}.png"
                mask_path = os.path.join(output_dir, mask_filename)
                cv2.imwrite(mask_path, mask_resized)
                logger.info(f"Individual mask saved: {mask_path}")
            
            # Save final combined mask
            final_mask_path = os.path.join(output_dir, "final_mask.png")
            cv2.imwrite(final_mask_path, final_mask)
            logger.info(f"Final mask saved: {final_mask_path}")
            
            # Resize for display
            width = int(img.shape[1] * scale_percent / 100)
            height = int(img.shape[0] * scale_percent / 100)
            dim = (width, height)
            
            resized_img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
            resized_mask = cv2.resize(final_mask, dim, interpolation=cv2.INTER_AREA)
            
            return final_mask, resized_img, resized_mask
            
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            raise
    
    def display_results(self, resized_img, resized_mask):
        """
        Display the results using OpenCV windows.
        
        Args:
            resized_img: Resized original image
            resized_mask: Resized mask
        """
        try:
            cv2.imshow("Original Image", resized_img)
            cv2.imshow("Final Mask", resized_mask)
            logger.info("Press any key to close windows...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        except Exception as e:
            logger.error(f"Display failed: {e}")

def main():
    """Main function to run hair segmentation prediction with resize."""
    # Configuration from config.py
    model_path = str(BEST_MODEL_PATH)
    image_path = str(TEST_IMAGES_DIR / "1621.jpg")
    output_dir = str(OUTPUT_DIR)
    scale_percent = 100  # 100 = original size
    
    # Check if model exists
    if not os.path.exists(model_path):
        logger.error(f"Model not found: {model_path}")
        logger.info("Please train the model first using train.py")
        return
    
    # Check if image exists
    if not os.path.exists(image_path):
        logger.error(f"Image not found: {image_path}")
        logger.info(f"Please ensure test image exists in {TEST_IMAGES_DIR}")
        return
    
    # Run prediction
    try:
        predictor = YOLOv8ResizePredictor(model_path)
        final_mask, resized_img, resized_mask = predictor.predict_with_resize(
            image_path, output_dir, scale_percent
        )
        
        # Display results
        predictor.display_results(resized_img, resized_mask)
        
        logger.info("Prediction completed successfully!")
        
    except Exception as e:
        logger.error(f"Prediction failed: {e}")

if __name__ == "__main__":
    main()
