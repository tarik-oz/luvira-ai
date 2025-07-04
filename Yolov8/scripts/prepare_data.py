"""
YOLOv8 Data Preparation Script

This script converts binary masks to YOLOv8 polygon format for training.
It processes mask images and converts them to the required annotation format.
"""

import os
import cv2
import logging
from pathlib import Path
from config import OUTPUT_DIR, TRAIN_LABELS_DIR

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class YOLOv8DataPreparator:
    """Data preparation utility for YOLOv8 training."""
    
    def __init__(self, input_dir: str = None, output_dir: str = None):
        """
        Initialize the data preparator.
        
        Args:
            input_dir: Directory containing mask images (uses config default if None)
            output_dir: Directory to save YOLOv8 annotations (uses config default if None)
        """
        self.input_dir = Path(input_dir) if input_dir else OUTPUT_DIR
        self.output_dir = Path(output_dir) if output_dir else TRAIN_LABELS_DIR
        self._setup_directories()
    
    def _setup_directories(self):
        """Create output directory if it doesn't exist."""
        try:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Output directory ready: {self.output_dir}")
        except Exception as e:
            logger.error(f"Failed to create output directory: {e}")
            raise
    
    def process_mask(self, mask_path: Path) -> list:
        """
        Process a single mask image and convert to polygons.
        
        Args:
            mask_path: Path to the mask image
            
        Returns:
            List of polygons in YOLOv8 format
        """
        try:
            # Load the binary mask
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if mask is None:
                logger.warning(f"Could not load mask: {mask_path}")
                return []
            
            # Apply binary threshold
            _, mask = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)
            
            H, W = mask.shape
            
            # Find contours
            contours, hierarchy = cv2.findContours(
                mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            
            # Convert contours to polygons
            polygons = []
            for cnt in contours:
                # Filter small contours
                if cv2.contourArea(cnt) > 200:
                    polygon = []
                    for point in cnt:
                        x, y = point[0]
                        # Normalize coordinates
                        polygon.append(x / W)
                        polygon.append(y / H)
                    polygons.append(polygon)
            
            return polygons
            
        except Exception as e:
            logger.error(f"Error processing mask {mask_path}: {e}")
            return []
    
    def save_annotation(self, polygons: list, output_name: str):
        """
        Save polygons to YOLOv8 annotation file.
        
        Args:
            polygons: List of polygons
            output_name: Name for the output file
        """
        try:
            output_path = self.output_dir / f"{output_name}.txt"
            
            with open(output_path, "w") as f:
                for polygon in polygons:
                    for p_idx, p in enumerate(polygon):
                        if p_idx == len(polygon) - 1:
                            f.write(f"{p}\n")
                        elif p_idx == 0:
                            f.write(f"0 {p} ")
                        else:
                            f.write(f"{p} ")
            
            logger.info(f"Annotation saved: {output_path}")
            
        except Exception as e:
            logger.error(f"Error saving annotation {output_name}: {e}")
    
    def process_all_masks(self):
        """Process all mask images in the input directory."""
        try:
            if not self.input_dir.exists():
                logger.error(f"Input directory does not exist: {self.input_dir}")
                return
            
            mask_files = list(self.input_dir.glob("*.png"))
            logger.info(f"Found {len(mask_files)} mask files to process")
            
            processed_count = 0
            
            for mask_path in mask_files:
                logger.info(f"Processing: {mask_path.name}")
                
                # Process mask
                polygons = self.process_mask(mask_path)
                
                if polygons:
                    # Generate output name
                    output_name = mask_path.stem.replace("_hair", "").lstrip("0")
                    
                    # Save annotation
                    self.save_annotation(polygons, output_name)
                    processed_count += 1
                else:
                    logger.warning(f"No valid polygons found in {mask_path.name}")
            
            logger.info(f"Processing completed. {processed_count} files processed successfully.")
            
        except Exception as e:
            logger.error(f"Error processing masks: {e}")
            raise

def main():
    """Main function to run data preparation."""
    try:
        # Configuration from config.py
        input_dir = str(OUTPUT_DIR)
        output_dir = str(TRAIN_LABELS_DIR)
        
        # Check if input directory exists
        if not os.path.exists(input_dir):
            logger.error(f"Input directory not found: {input_dir}")
            logger.info(f"Please ensure mask images are in {OUTPUT_DIR}")
            return
        
        # Create preparator and process
        preparator = YOLOv8DataPreparator(input_dir, output_dir)
        preparator.process_all_masks()
        
        logger.info("Data preparation completed successfully!")
        
    except Exception as e:
        logger.error(f"Data preparation failed: {e}")

if __name__ == "__main__":
    main()
