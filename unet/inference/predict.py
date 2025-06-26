#!/usr/bin/env python3
"""
Main prediction script for hair segmentation U-Net model.
This script handles model inference and result visualization.
"""

import sys
import argparse
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from inference.predictor import create_predictor
from training.trainer import create_trainer
from config import BEST_MODEL_PATH
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_model(model_path: Path = None):
    """
    Load the trained model.
    
    Args:
        model_path: Path to the model file
        
    Returns:
        Loaded model
    """
    if model_path is None:
        model_path = BEST_MODEL_PATH
    
    if not Path(model_path).exists():
        raise FileNotFoundError(f"Model not found at {model_path}")
    
    logger.info(f"Loading model from {model_path}")
    
    # Create trainer to load model
    trainer = create_trainer()
    model = trainer.load_trained_model(model_path)
    
    return model


def predict_single_image(image_path: str, model_path: Path = None):
    """
    Predict segmentation for a single image.
    
    Args:
        image_path: Path to the input image
        model_path: Path to the model file
    """
    try:
        # Load model
        model = load_model(model_path)
        
        # Create predictor
        predictor = create_predictor(model)
        
        # Make prediction
        original_image, predicted_mask, binary_mask = predictor.predict(image_path)
        
        if original_image is None:
            logger.error("Failed to process image")
            return
        
        # Visualize results
        predictor.visualize_prediction(original_image, predicted_mask, binary_mask)
        
        # Save results
        output_name = Path(image_path).stem
        predictor.predict_and_save(image_path, output_name)
        
        logger.info(f"Prediction completed for {image_path}")
        
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise


def predict_directory(input_dir: str, model_path: Path = None):
    """
    Predict segmentation for all images in a directory.
    
    Args:
        input_dir: Path to the input directory
        model_path: Path to the model file
    """
    try:
        # Load model
        model = load_model(model_path)
        
        # Create predictor
        predictor = create_predictor(model)
        
        # Predict all images in directory
        results = predictor.predict_directory(Path(input_dir))
        
        successful = sum(results)
        total = len(results)
        logger.info(f"Directory prediction completed: {successful}/{total} successful")
        
    except Exception as e:
        logger.error(f"Directory prediction failed: {e}")
        raise


def main():
    """
    Main prediction function.
    """
    parser = argparse.ArgumentParser(description="Hair Segmentation Prediction")
    parser.add_argument(
        "--input", 
        type=str, 
        required=True,
        help="Path to input image or directory"
    )
    parser.add_argument(
        "--model", 
        type=str, 
        default=str(BEST_MODEL_PATH),
        help="Path to trained model"
    )
    parser.add_argument(
        "--mode", 
        type=str, 
        choices=["single", "directory"], 
        default="single",
        help="Prediction mode: single image or directory"
    )
    
    args = parser.parse_args()
    
    try:
        if args.mode == "single":
            logger.info(f"Predicting single image: {args.input}")
            predict_single_image(args.input, Path(args.model))
        else:
            logger.info(f"Predicting directory: {args.input}")
            predict_directory(args.input, Path(args.model))
            
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise


if __name__ == "__main__":
    main() 