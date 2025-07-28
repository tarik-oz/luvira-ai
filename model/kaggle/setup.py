#!/usr/bin/env python3
"""
Kaggle Setup Script for Hair Segmentation Project
This script helps prepare and verify the Kaggle environment for training.
"""

import sys
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class KaggleSetup:
    """
    Kaggle environment setup and verification class.
    """
    
    def __init__(self):
        self.kaggle_input = Path("/kaggle/input")
        self.kaggle_working = Path("/kaggle/working")
        self.model_dir = self.kaggle_working / "model"
        self.dataset_name = "hair-dataset-30k"  # Change this to match your dataset name
        
    def check_environment(self):
        """Check if we're running in Kaggle environment."""
        logger.info("Checking Kaggle environment...")
        
        if not self.kaggle_input.exists():
            logger.error("Not running in Kaggle environment!")
            return False
            
        if not self.kaggle_working.exists():
            logger.error("Kaggle working directory not found!")
            return False
            
        logger.info("Kaggle environment detected")
        return True
    
    def check_dataset(self):
        """Check if dataset is properly attached."""
        logger.info(f"Checking dataset: {self.dataset_name}")
        
        dataset_path = self.kaggle_input / self.dataset_name
        
        if not dataset_path.exists():
            logger.error(f"Dataset not found: {dataset_path}")
            logger.info("Available datasets:")
            for item in self.kaggle_input.iterdir():
                logger.info(f"  - {item.name}")
            return False
            
        # Check for required directories
        images_dir = dataset_path / "images"
        masks_dir = dataset_path / "masks"
        
        if not images_dir.exists():
            logger.error(f"Images directory not found: {images_dir}")
            return False
            
        if not masks_dir.exists():
            logger.error(f"Masks directory not found: {masks_dir}")
            return False
            
        # Count files
        image_count = len(list(images_dir.glob("*.*")))
        mask_count = len(list(masks_dir.glob("*.*")))
        
        logger.info(f"Dataset found with {image_count} images and {mask_count} masks")
        return True
    
    def check_gpu(self):
        """Check GPU availability."""
        logger.info("Checking GPU availability...")
        
        try:
            import torch
            
            if not torch.cuda.is_available():
                logger.warning("⚠️  GPU not available, using CPU")
                return False
            
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            
            logger.info(f"✅ GPU Available: {gpu_name}")
            logger.info(f"✅ GPU Memory: {gpu_memory:.2f} GB")
            
            return True
            
        except ImportError:
            logger.error("❌ PyTorch not available!")
            return False
    
    def install_dependencies(self):
        """Install required dependencies."""
        logger.info("Installing dependencies...")
        
        try:
            import subprocess
            
            # Install OpenCV
            subprocess.run([sys.executable, "-m", "pip", "install", "opencv-python-headless"], 
                          check=True, capture_output=True)
            logger.info("✅ OpenCV installed")
            
            # Install tqdm if not available
            try:
                import tqdm
            except ImportError:
                subprocess.run([sys.executable, "-m", "pip", "install", "tqdm"], 
                              check=True, capture_output=True)
                logger.info("✅ tqdm installed")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to install dependencies: {e}")
            return False
    
    def create_directory_structure(self):
        """Create necessary directories."""
        logger.info("Creating directory structure...")
        
        directories = [
            self.kaggle_working / "trained_models",
            self.kaggle_working / "logs",
            self.kaggle_working / "processed"
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
            logger.info(f"✅ Created: {directory}")
        
        return True
    
    def run_setup(self):
        """Run complete setup process."""
        logger.info("=" * 60)
        logger.info("KAGGLE SETUP FOR HAIR SEGMENTATION")
        logger.info("=" * 60)
        
        steps = [
            ("Environment Check", self.check_environment),
            ("Dataset Check", self.check_dataset),
            ("GPU Check", self.check_gpu),
            ("Install Dependencies", self.install_dependencies),
            ("Create Directories", self.create_directory_structure)
        ]
        
        for step_name, step_func in steps:
            logger.info(f"\n🔄 {step_name}...")
            try:
                success = step_func()
                if not success and step_name in ["Environment Check", "Dataset Check"]:
                    logger.error(f"❌ {step_name} failed - setup cannot continue")
                    return False
                elif not success:
                    logger.warning(f"⚠️  {step_name} completed with warnings")
            except Exception as e:
                logger.error(f"❌ {step_name} failed with error: {e}")
                if step_name in ["Environment Check", "Dataset Check"]:
                    return False
        
        logger.info("\n" + "=" * 60)
        logger.info("SETUP COMPLETE!")
        logger.info("=" * 60)
        logger.info("✅ Ready for training!")
        logger.info("Run: python main_kaggle_train.py")
        logger.info("=" * 60)
        
        return True

def main():
    """Main setup function."""
    setup = KaggleSetup()
    
    # Run setup
    success = setup.run_setup()
    
    if success:
        logger.info("\n🎉 Setup completed successfully!")
        logger.info("You can now start training your hair segmentation model!")
    else:
        logger.error("\n❌ Setup failed!")
        logger.error("Please check the errors above and fix them before continuing.")

if __name__ == "__main__":
    main() 