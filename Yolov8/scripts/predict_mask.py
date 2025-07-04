"""
YOLOv8 Hair Segmentation Prediction Script

This script performs hair segmentation using a trained YOLOv8 model.
"""

from ultralytics import YOLO
import os
import cv2
import matplotlib.pyplot as plt
import logging
from pathlib import Path
from config import BEST_MODEL_PATH, TEST_IMAGES_DIR, OUTPUT_DIR

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class YOLOv8HairPredictor:
    """YOLOv8-based hair segmentation predictor."""
    
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
    
    def predict_mask(self, image_path: str, output_dir: str = None):
        """
        Predict hair mask for an image.
        
        Args:
            image_path: Path to input image
            output_dir: Directory to save results (uses config default if None)
            
        Returns:
            List of predicted masks
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
            
            # Get image name for output
            image_name = Path(image_path).stem
            
            # Run prediction
            results = self.model(img)
            
            # Process results
            masks = []
            fig, axes = plt.subplots(1, len(results) + 1, figsize=(15, 5))
            
            # Plot original image
            axes[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            axes[0].set_title("Original Image")
            axes[0].axis("off")
            
            # Process each detection
            for i, result in enumerate(results):
                for j, mask in enumerate(result.masks.data):
                    # Convert mask to numpy and scale
                    mask_np = mask.cpu().numpy() * 255
                    
                    # Resize to original image size
                    mask_resized = cv2.resize(mask_np, (img_width, img_height))
                    masks.append(mask_resized)
                    
                    # Plot mask
                    axes[i + 1].imshow(mask_resized, cmap="gray")
                    axes[i + 1].set_title(f"Mask {j+1}")
                    axes[i + 1].axis("off")
                    
                    # Save mask
                    mask_path = os.path.join(output_dir, f"{image_name}_mask.png")
                    cv2.imwrite(mask_path, mask_resized)
                    logger.info(f"Mask saved to: {mask_path}")
            
            # Save visualization
            viz_path = os.path.join(output_dir, f"{image_name}_visualization.png")
            plt.tight_layout()
            plt.savefig(viz_path, dpi=300, bbox_inches='tight')
            plt.show()
            
            logger.info(f"Visualization saved to: {viz_path}")
            return masks
            
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            raise

def main():
    """Main function to run hair segmentation prediction."""
    # Configuration from config.py
    model_path = str(BEST_MODEL_PATH)
    image_path = str(TEST_IMAGES_DIR / "1624.jpg")
    output_dir = str(OUTPUT_DIR)
    
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
        predictor = YOLOv8HairPredictor(model_path)
        masks = predictor.predict_mask(image_path, output_dir)
        logger.info(f"Prediction completed. Generated {len(masks)} masks.")
    except Exception as e:
        logger.error(f"Prediction failed: {e}")

if __name__ == "__main__":
    main()
