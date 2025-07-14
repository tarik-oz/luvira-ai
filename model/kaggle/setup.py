#!/usr/bin/env python3
"""
Kaggle Setup Script for Hair Segmentation Project
This script helps prepare and verify the Kaggle environment for training.
"""

import os
import sys
import shutil
import logging
from pathlib import Path
import zipfile

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
        self.dataset_name = "hair_dataset_30k"  # Change this to match your dataset name
        
    def check_environment(self):
        """Check if we're running in Kaggle environment."""
        logger.info("Checking Kaggle environment...")
        
        if not self.kaggle_input.exists():
            logger.error("❌ Not running in Kaggle environment!")
            return False
            
        if not self.kaggle_working.exists():
            logger.error("❌ Kaggle working directory not found!")
            return False
            
        logger.info("✅ Kaggle environment detected")
        return True
    
    def check_dataset(self):
        """Check if dataset is properly attached."""
        logger.info(f"Checking dataset: {self.dataset_name}")
        
        dataset_path = self.kaggle_input / self.dataset_name
        
        if not dataset_path.exists():
            logger.error(f"❌ Dataset not found: {dataset_path}")
            logger.error("Please add your dataset to this notebook")
            return False
        
        # Check images folder
        images_path = dataset_path / "images"
        if not images_path.exists():
            logger.error(f"❌ Images folder not found: {images_path}")
            return False
        
        # Check masks folder
        masks_path = dataset_path / "masks"
        if not masks_path.exists():
            logger.error(f"❌ Masks folder not found: {masks_path}")
            return False
        
        # Count files
        image_files = list(images_path.glob("*.jpg")) + list(images_path.glob("*.png"))
        mask_files = list(masks_path.glob("*.webp")) + list(masks_path.glob("*.png"))
        
        logger.info(f"✅ Found {len(image_files)} images and {len(mask_files)} masks")
        
        if len(image_files) == 0 or len(mask_files) == 0:
            logger.error("❌ No image or mask files found!")
            return False
            
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
            self.model_dir,
            self.model_dir / "training",
            self.model_dir / "data_loader",
            self.model_dir / "models",
            self.model_dir / "utils",
            self.kaggle_working / "trained_models",
            self.kaggle_working / "logs"
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
            logger.info(f"✅ Created: {directory}")
        
        return True
    
    def verify_files(self):
        """Verify all required files are present."""
        logger.info("Verifying required files...")
        
        required_files = [
            "config_kaggle.py",
            "train_kaggle.py",
            "training/trainer_kaggle.py",
            "training/callbacks.py",
            "training/metrics.py",
            "data_loader/base_data_loader.py",
            "data_loader/lazy_data_loader.py",
            "data_loader/kaggle_data_loader.py",
            "data_loader/factory_kaggle.py",
            "models/unet_model.py",
            "models/attention_unet_model.py",
            "utils/model_saving.py",
            "utils/data_timestamp.py"
        ]
        
        missing_files = []
        
        for file_path in required_files:
            full_path = self.model_dir / file_path
            if not full_path.exists():
                missing_files.append(file_path)
                logger.error(f"❌ Missing: {file_path}")
            else:
                logger.info(f"✅ Found: {file_path}")
        
        if missing_files:
            logger.error(f"❌ Missing {len(missing_files)} required files!")
            return False
        
        logger.info("✅ All required files found")
        return True
    
    def create_init_files(self):
        """Create __init__.py files for Python imports."""
        logger.info("Creating __init__.py files...")
        
        init_files = [
            self.model_dir / "__init__.py",
            self.model_dir / "training" / "__init__.py",
            self.model_dir / "data_loader" / "__init__.py",
            self.model_dir / "models" / "__init__.py",
            self.model_dir / "utils" / "__init__.py"
        ]
        
        for init_file in init_files:
            if not init_file.exists():
                init_file.write_text("# Auto-generated __init__.py\n")
                logger.info(f"✅ Created: {init_file}")
        
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
            ("Create Directories", self.create_directory_structure),
            ("Verify Files", self.verify_files),
            ("Create Init Files", self.create_init_files)
        ]
        
        for step_name, step_func in steps:
            logger.info(f"\n🔄 {step_name}...")
            try:
                success = step_func()
                if not success and step_name in ["Environment Check", "Dataset Check", "Verify Files"]:
                    logger.error(f"❌ {step_name} failed - setup cannot continue")
                    return False
                elif not success:
                    logger.warning(f"⚠️  {step_name} completed with warnings")
            except Exception as e:
                logger.error(f"❌ {step_name} failed with error: {e}")
                if step_name in ["Environment Check", "Dataset Check", "Verify Files"]:
                    return False
        
        logger.info("\n" + "=" * 60)
        logger.info("SETUP COMPLETE!")
        logger.info("=" * 60)
        logger.info("✅ Ready for training!")
        logger.info("Run: python /kaggle/working/model/train_kaggle.py")
        logger.info("=" * 60)
        
        return True
    
    def generate_quickstart_notebook(self):
        """Generate a quickstart notebook code."""
        notebook_code = '''
# Kaggle Hair Segmentation - Quick Start
# =====================================

# 1. Setup environment
import os
import sys
sys.path.append('/kaggle/working/model')

# 2. Run setup
from kaggle_setup import KaggleSetup
setup = KaggleSetup()
setup.run_setup()

# 3. Start training
os.chdir('/kaggle/working/model')
!python train_kaggle.py

# 4. Check results
print("Training completed!")
print("Model saved to /kaggle/working/trained_models/")
'''
        
        quickstart_path = self.kaggle_working / "quickstart_notebook.py"
        quickstart_path.write_text(notebook_code)
        
        logger.info(f"✅ Quickstart notebook saved to: {quickstart_path}")
        return quickstart_path

def main():
    """Main setup function."""
    setup = KaggleSetup()
    
    # Run setup
    success = setup.run_setup()
    
    if success:
        # Generate quickstart notebook
        setup.generate_quickstart_notebook()
        
        logger.info("\n🎉 Setup completed successfully!")
        logger.info("You can now start training your hair segmentation model!")
    else:
        logger.error("\n❌ Setup failed!")
        logger.error("Please check the errors above and fix them before continuing.")

if __name__ == "__main__":
    main() 