#!/usr/bin/env python3
"""
Main prediction script for hair segmentation U-Net model.
This script handles model inference and result visualization.
"""

import sys
import argparse
from pathlib import Path
from typing import Optional, List, Tuple

# Add project root to path so that we can use absolute imports
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))

from model.inference.predictor import create_predictor, HairSegmentationPredictor
from model.training.trainer import create_trainer
from model.config import DEFAULT_MODEL_PATH, TEST_IMAGES_DIR, TEST_RESULTS_DIR
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_arguments():
    """
    Parse command line arguments.
    
    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(description="Hair Segmentation Prediction")
    parser.add_argument(
        "--images",
        nargs="+",
        type=str,
        help="Space-separated list of image names to process from images-dir"
    )
    parser.add_argument(
        "--images-dir",
        type=str,
        help="Directory containing images (uses TEST_IMAGES_DIR from config if not provided)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=str(DEFAULT_MODEL_PATH),
        help="Path to trained model (uses DEFAULT_MODEL_PATH from config if not provided)"
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Device to use for prediction"
    )
    parser.add_argument(
        "--no-visualization",
        action="store_true",
        help="Disable visualization plots"
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        help="Directory to save results (uses TEST_RESULTS_DIR from config if not provided)"
    )
    
    return parser.parse_args()


def load_model(model_path: Path):
    """
    Load the trained model.
    
    Args:
        model_path: Path to the model file (required)
        
    Returns:
        Loaded model
    """
    # Convert model_path to Path object if it's a string
    if isinstance(model_path, str):
        model_path = Path(model_path)
        
    # Check if model file exists
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found at {model_path}")
    
    logger.info(f"Loading model from {model_path}")
    
    # Create trainer to load model
    trainer = create_trainer()
    model, config = trainer.load_trained_model(model_path)
    
    # Log model type information from config
    model_type = None
    if config is not None:
        model_type = config.get("model_config", {}).get("model_type", "unknown")
    if not model_type:
        model_type = "unknown"
    logger.info(f"Model type: {model_type}")
    
    return model


def validate_images(temp_predictor: HairSegmentationPredictor, input_dir: Path, image_names: Optional[List[str]] = None):
    """
    Validate input directory and images.
    
    Args:
        temp_predictor: Temporary predictor instance for validation
        input_dir: Directory to validate
        image_names: Optional list of specific image names to validate
    """
    if image_names:
        temp_predictor.validate_input_directory(input_dir)
        temp_predictor.validate_image_names(image_names, input_dir)
    else:
        temp_predictor.validate_input_directory(input_dir)
        temp_predictor.find_valid_images(input_dir)


def process_images(predictor: HairSegmentationPredictor, input_dir: Path, 
                  show_visualization: bool, image_names: Optional[List[str]] = None) -> Tuple[int, int]:
    """
    Process images using the predictor.
    
    Args:
        predictor: Predictor instance
        input_dir: Directory containing images
        show_visualization: Whether to show visualization plots
        image_names: Optional list of specific image names to process
        
    Returns:
        Tuple of (successful_count, total_count)
    """
    if image_names:
        results = predictor.predict_batch(
            image_names,
            show_visualization=show_visualization,
            input_dir=input_dir
        )
    else:
        results = predictor.predict_directory(
            input_dir,
            show_visualization=show_visualization
        )
        
    return sum(results), len(results)


def main():
    """
    Main prediction function.
    """
    try:
        # Parse arguments
        args = parse_arguments()
        
        # Get input and output directories
        input_dir = Path(args.images_dir) if args.images_dir else TEST_IMAGES_DIR
        results_dir = Path(args.results_dir) if args.results_dir else TEST_RESULTS_DIR
        
        # Create temporary predictor just for validation (without loading model)
        temp_predictor = HairSegmentationPredictor(
            None,  # No model needed for validation
            test_images_dir=input_dir,
            test_results_dir=results_dir,
            device="cpu"  # Use CPU for validation since no model operations
        )
        
        # Validate images first
        validate_images(temp_predictor, input_dir, args.images)
        
        # If validation passed, load model
        model = load_model(args.model)
        
        # Create real predictor with model
        predictor = create_predictor(
            model, 
            device=args.device,
            test_images_dir=input_dir,
            test_results_dir=results_dir
        )
        
        # Process images
        successful, total = process_images(
            predictor,
            input_dir,
            not args.no_visualization,
            args.images
        )
        
        # Log results
        logger.info(f"Prediction completed: {successful}/{total} successful")
            
    except ValueError as e:
        # Extract missing images from error message if present
        error_msg = str(e)
        if "Images not found" in error_msg:
            # Try to extract missing images from the error
            try:
                missing_images = error_msg.split(": ")[-1]
                error_msg = f"Following images not found in {input_dir}:\n{missing_images}"
            except:
                error_msg = f"Image(s) not found in directory: {input_dir}"
        elif "No valid images found" in error_msg:
            error_msg = f"No valid images found in directory: {input_dir}"
        elif "directory not found" in error_msg:
            error_msg = f"Directory not found: {input_dir}"
            
        logger.error(error_msg)
        sys.exit(1)
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise


if __name__ == "__main__":
    main() 