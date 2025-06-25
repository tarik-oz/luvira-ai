#!/usr/bin/env python3
"""
Test script to verify PyTorch setup for hair segmentation project.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

import torch
import torch.nn as nn
import numpy as np
import logging

from models.unet_model import create_unet_model
from config import MODEL_CONFIG

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_pytorch_installation():
    """Test PyTorch installation and GPU availability."""
    logger.info("Testing PyTorch installation...")
    
    # Check PyTorch version
    logger.info(f"PyTorch version: {torch.__version__}")
    
    # Check CUDA availability
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"CUDA version: {torch.version.cuda}")
        logger.info(f"Number of GPUs: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            logger.info(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    
    # Test device selection
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    return device


def test_model_creation():
    """Test U-Net model creation."""
    logger.info("Testing U-Net model creation...")
    
    try:
        # Create model
        model = create_unet_model()
        logger.info("Model created successfully")
        
        # Print model summary
        model.summary()
        
        # Test forward pass
        batch_size = 2
        input_shape = MODEL_CONFIG["input_shape"]
        test_input = torch.randn(batch_size, *input_shape)
        
        logger.info(f"Test input shape: {test_input.shape}")
        
        # Move to device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        test_input = test_input.to(device)
        
        # Forward pass
        with torch.no_grad():
            output = model(test_input)
        
        logger.info(f"Model output shape: {output.shape}")
        logger.info("Forward pass successful")
        
        return True
        
    except Exception as e:
        logger.error(f"Model creation failed: {e}")
        return False


def test_data_loading():
    """Test data loading functionality."""
    logger.info("Testing data loading...")
    
    try:
        from data.data_loader import create_data_loader
        
        # Create data loader
        data_loader = create_data_loader()
        logger.info("Data loader created successfully")
        
        # Check if data directories exist
        from config import IMAGES_DIR, MASKS_DIR
        logger.info(f"Images directory exists: {IMAGES_DIR.exists()}")
        logger.info(f"Masks directory exists: {MASKS_DIR.exists()}")
        
        return True
        
    except Exception as e:
        logger.error(f"Data loading test failed: {e}")
        return False


def main():
    """Main test function."""
    logger.info("Starting PyTorch setup tests...")
    
    # Test PyTorch installation
    device = test_pytorch_installation()
    
    # Test model creation
    model_success = test_model_creation()
    
    # Test data loading
    data_success = test_data_loading()
    
    # Summary
    logger.info("\n" + "="*50)
    logger.info("TEST SUMMARY")
    logger.info("="*50)
    logger.info(f"PyTorch installation: ✓")
    logger.info(f"Model creation: {'✓' if model_success else '✗'}")
    logger.info(f"Data loading: {'✓' if data_success else '✗'}")
    logger.info(f"Device: {device}")
    
    if model_success and data_success:
        logger.info("\nAll tests passed! PyTorch setup is ready.")
    else:
        logger.error("\nSome tests failed. Please check the errors above.")


if __name__ == "__main__":
    main() 