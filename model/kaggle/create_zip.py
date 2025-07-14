#!/usr/bin/env python3
"""
Kaggle Zip Creator for Hair Segmentation Project - ORGANIZED VERSION
This script creates a zip file with all necessary files for Kaggle training.
All files are now organized in model/kaggle/ directory.
"""

import os
import zipfile
import logging
import re
from pathlib import Path
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def fix_imports(file_content, filename=""):
    """
    Fix import statements in file content for Kaggle compatibility.
    
    Args:
        file_content: The content of the file to fix
        filename: Optional filename for special case handling
    
    Returns:
        Fixed content with proper imports
    """
    # Replace relative imports with absolute imports
    content = file_content
    
    # Special handling for kaggle directory files
    if "model/kaggle/" in filename or filename.startswith(("data_loader.py", "factory.py", "trainer.py", "train.py")):
        # Fix relative import patterns in kaggle files
        
        # Handle importing from parent directory modules
        content = re.sub(r'from \.\.(base_data_loader|lazy_data_loader|traditional_data_loader)', 
                        r'from model.data_loader.\1', content)
        
        # Handle local imports within the kaggle directory
        content = re.sub(r'from \.([a-zA-Z_]+) import', 
                        r'from model.kaggle.\1 import', content)
        
        # Fix direct imports that should be prefixed with model.
        content = re.sub(r'from (base_data_loader|lazy_data_loader|traditional_data_loader) import', 
                        r'from model.data_loader.\1 import', content)
        
        # Fix direct imports within the kaggle directory
        content = re.sub(r'\nfrom (config|factory|data_loader|trainer|train|callbacks|metrics|lazy_dataset) import', 
                        r'\nfrom model.kaggle.\1 import', content)
        
        # Fix data_loader import from kaggle directory
        content = re.sub(r'from data_loader import', 
                        r'from model.kaggle.data_loader import', content)
        
        # Fix factory import from kaggle directory
        content = re.sub(r'from factory import', 
                        r'from model.kaggle.factory import', content)
        
        # Fix lazy_dataset import from kaggle directory
        content = re.sub(r'from lazy_dataset import', 
                        r'from model.kaggle.lazy_dataset import', content)
    
    # Fix model files import patterns (more specific patterns first)
    if "class UNetModel" in content or "class AttentionUNetModel" in content:
        # For model files, we need more specific fixes
        content = re.sub(r'try:\s+from \.\.(config|kaggle\.config) import MODEL_CONFIG', 
                         r'try:\n    from model.kaggle.config import MODEL_CONFIG', content)
        content = re.sub(r'except ImportError:\s+from config import MODEL_CONFIG', 
                         r'except ImportError:\n    from model.kaggle.config import MODEL_CONFIG', content)
    
    # General import fixes
    content = re.sub(r'from \.\.(config|utils|models|training)', r'from model.\1', content)
    content = re.sub(r'from \.\.(data_loader)', r'from model.data_loader', content)
    content = re.sub(r'from config import', r'from model.kaggle.config import', content)
    content = re.sub(r'from \.\.(kaggle)', r'from model.kaggle', content)
    
    # More specific fixes for direct imports
    content = re.sub(r'from models\.', r'from model.models.', content)
    content = re.sub(r'from utils\.', r'from model.utils.', content)
    content = re.sub(r'from data_loader\.', r'from model.data_loader.', content)
    content = re.sub(r'from training\.', r'from model.training.', content)
    
    return content

def create_kaggle_zip():
    """
    Create a zip file with all necessary Kaggle files from model/kaggle/ directory.
    """
    logger.info("Creating Kaggle training zip file from organized structure...")
    
    # Get the current directory (should be model/kaggle/)
    current_dir = Path(__file__).parent
    project_root = current_dir.parent.parent  # Go up to project root
    
    # Create zip filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    zip_filename = f"hair_segmentation_kaggle_{timestamp}.zip"
    zip_path = project_root / zip_filename
    
    # Required files from kaggle directory
    kaggle_files = [
        "config.py",
        "trainer.py", 
        "train.py",
        "setup.py",
        "data_loader.py",
        "factory.py",
        "lazy_dataset.py",  # Add our new file with augmentations
        "__init__.py"
    ]
    
    # Required files from parent directories
    external_files = [
        ("model/models/unet_model.py", "model/models/unet_model.py"),
        ("model/models/attention_unet_model.py", "model/models/attention_unet_model.py"),
        ("model/data_loader/base_data_loader.py", "model/data_loader/base_data_loader.py"),
        ("model/data_loader/lazy_data_loader.py", "model/data_loader/lazy_data_loader.py"),
        ("model/data_loader/traditional_data_loader.py", "model/data_loader/traditional_data_loader.py"),
        ("model/data_loader/lazy_dataset.py", "model/data_loader/lazy_dataset.py"),
        ("model/training/callbacks.py", "model/kaggle/callbacks.py"),
        ("model/training/metrics.py", "model/kaggle/metrics.py"),
        ("model/utils/model_saving.py", "model/utils/model_saving.py"),
        ("model/utils/data_timestamp.py", "model/utils/data_timestamp.py"),
        ("requirements.txt", "requirements.txt")
    ]
    
    try:
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            
            # Add files from kaggle directory
            for filename in kaggle_files:
                file_path = current_dir / filename
                if file_path.exists():
                    # Read and fix imports before adding to zip
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                    fixed_content = fix_imports(content, f"model/kaggle/{filename}")
                    zipf.writestr(f"model/kaggle/{filename}", fixed_content)
                    logger.info(f"[+] Added (with import fixes): model/kaggle/{filename}")
                else:
                    logger.warning(f"[-] Missing: {file_path}")
            
            # Add external required files
            for source_path, zip_path_in_archive in external_files:
                full_source_path = project_root / source_path
                if full_source_path.exists():
                    # Read and fix imports before adding to zip
                    with open(full_source_path, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                    fixed_content = fix_imports(content, zip_path_in_archive)
                    zipf.writestr(zip_path_in_archive, fixed_content)
                    logger.info(f"[+] Added (with import fixes): {zip_path_in_archive}")
                else:
                    logger.warning(f"[-] Missing: {full_source_path}")
            
            # Create __init__.py files for each directory to ensure proper imports
            init_dirs = ["model", "model/models", "model/data_loader", "model/training", "model/utils"]
            for init_dir in init_dirs:
                zipf.writestr(f"{init_dir}/__init__.py", "# Module initialization\n")
                logger.info(f"[+] Added: {init_dir}/__init__.py")
            
            # Add a main training script that uses the kaggle module
            main_training_script = '''#!/usr/bin/env python3
"""
Main Kaggle training script for hair segmentation.
This script imports from the organized kaggle module.
"""

import sys
import os
import traceback
import logging
from pathlib import Path

# Add model directory to Python path
sys.path.append('/kaggle/working')

# Import and run the Kaggle training
if __name__ == "__main__":
    try:
        from model.kaggle.train import main
        print("Starting Hair Segmentation Training on Kaggle...")
        main()
    except Exception as e:
        print(f"Error during training: {e}")
        import traceback
        traceback.print_exc()
'''
            zipf.writestr("main_kaggle_train.py", main_training_script)
            logger.info(f"[+] Added: main_kaggle_train.py")
            
            # Add README for Kaggle usage
            readme_content = '''# Hair Segmentation - Kaggle Training

## Quick Start

1. Upload this zip file to Kaggle as a dataset
2. In your Kaggle notebook, extract it:
   ```python
   import zipfile
   with zipfile.ZipFile('/kaggle/input/your-dataset-name/hair_segmentation_kaggle_*.zip', 'r') as zipf:
       zipf.extractall('/kaggle/working/')
   ```

3. Run the training:
   ```python
   %cd /kaggle/working
   !python main_kaggle_train.py
   ```

## File Structure

```
model/
├── kaggle/              # Main Kaggle module
│   ├── config.py        # Kaggle configuration  
│   ├── trainer.py       # Training logic
│   ├── train.py         # Main training function
│   ├── data_loader.py   # Kaggle data loader
│   ├── factory.py       # Data loader factory
│   ├── lazy_dataset.py  # Enhanced dataset with augmentations
│   ├── callbacks.py     # Loss functions and callbacks
│   ├── metrics.py       # Evaluation metrics
│   └── setup.py         # Environment setup
├── models/              # Model architectures
├── data_loader/         # Base data loading
├── training/            # Training utilities
└── utils/               # Helper utilities
```

## Dataset Requirements

Your Kaggle dataset should be named according to config.py and contain:
- images/ folder with JPG files
- masks/ folder with WebP files

## Configuration

All settings are in `model/kaggle/config.py`:
- Batch size: 24 (optimized for P100)
- Epochs: 50
- Model: Attention U-Net
- Validation split: 0.15
- Data Augmentation: Enabled (HorizontalFlip, RandomRotation, ColorJitter, RandomBrightnessContrast, ElasticTransform)

The configuration is optimized for Kaggle P100 GPU.
'''
            zipf.writestr("README_KAGGLE.md", readme_content)
            logger.info(f"[+] Added: README_KAGGLE.md")
    
        logger.info(f"\n✅ Zip file created successfully: {zip_filename}")
        logger.info(f"📍 Location: {zip_path}")
        return zip_filename
        
    except Exception as e:
        logger.error(f"❌ Error creating zip file: {e}")
        return None

def create_extraction_script():
    """
    Create a script to extract the zip file in Kaggle.
    """
    extraction_script = '''#!/usr/bin/env python3
"""
Kaggle Extraction Script for Hair Segmentation Training
Run this in your Kaggle notebook to extract and setup the training environment.
"""

import zipfile
import os
from pathlib import Path

def extract_kaggle_files():
    """Extract the hair segmentation files in Kaggle."""
    
    print("🚀 Hair Segmentation - Kaggle Setup")
    print("=" * 50)
    
    # Find the dataset zip file
    dataset_path = "/kaggle/input"
    zip_file = None
    
    # Look for the zip file in input datasets
    for root, dirs, files in os.walk(dataset_path):
        for file in files:
            if file.startswith("hair_segmentation_kaggle_") and file.endswith(".zip"):
                zip_file = os.path.join(root, file)
                break
        if zip_file:
            break
    
    if not zip_file:
        print("❌ Could not find hair segmentation zip file in /kaggle/input/")
        print("Please make sure you uploaded the zip file as a dataset.")
        return False
    
    print(f"📦 Found zip file: {zip_file}")
    
    # Extract to working directory
    try:
        with zipfile.ZipFile(zip_file, 'r') as zipf:
            zipf.extractall("/kaggle/working/")
        
        print("✅ Files extracted to /kaggle/working/")
        
        # Verify key files exist
        key_files = [
            "/kaggle/working/model/kaggle/config.py",
            "/kaggle/working/model/kaggle/trainer.py", 
            "/kaggle/working/model/kaggle/train.py",
            "/kaggle/working/model/kaggle/callbacks.py",
            "/kaggle/working/model/kaggle/metrics.py",
            "/kaggle/working/model/kaggle/lazy_dataset.py",  # Check for our augmentation file
            "/kaggle/working/main_kaggle_train.py"
        ]
        
        missing_files = []
        for file in key_files:
            if not Path(file).exists():
                missing_files.append(file)
        
        if missing_files:
            print("⚠️  Some files are missing:")
            for file in missing_files:
                print(f"   - {file}")
            return False
        
        print("✅ All key files verified")
        print("\n🎯 Ready to start training!")
        print("Run: !python /kaggle/working/main_kaggle_train.py")
        
        return True
        
    except Exception as e:
        print(f"❌ Error extracting files: {e}")
        return False

if __name__ == "__main__":
    success = extract_kaggle_files()
    if success:
        print("\n" + "="*50)
        print("🚀 READY FOR TRAINING!")
        print("="*50)
    else:
        print("\n" + "="*50) 
        print("❌ SETUP FAILED - Please check the errors above")
        print("="*50)
'''
    
    with open("kaggle_extraction_script.py", "w", encoding='utf-8') as f:
        f.write(extraction_script)
    
    logger.info("[+] Created kaggle_extraction_script.py")

def main():
    """
    Main function to create Kaggle zip and related files.
    """
    logger.info("HAIR SEGMENTATION KAGGLE ZIP CREATOR - ORGANIZED VERSION")
    logger.info("=" * 60)
    
    # Change to the script directory
    script_dir = Path(__file__).parent
    os.chdir(script_dir)
    
    # Create the zip file
    zip_filename = create_kaggle_zip()
    
    if zip_filename:
        # Create extraction script
        create_extraction_script()
        
        logger.info("\n" + "="*60)
        logger.info("📋 KAGGLE UPLOAD INSTRUCTIONS")
        logger.info("="*60)
        logger.info("1. DATASET UPLOAD:")
        logger.info(f"   - Go to kaggle.com/datasets")
        logger.info(f"   - Create new dataset")
        logger.info(f"   - Upload: {zip_filename}")
        logger.info(f"   - Make it public")
        logger.info("")
        logger.info("2. KAGGLE NOTEBOOK:")
        logger.info("   - Create new notebook with GPU enabled")
        logger.info("   - Add your dataset + hairdataset")  
        logger.info("   - Upload and run: kaggle_extraction_script.py")
        logger.info("   - Then run: !python /kaggle/working/main_kaggle_train.py")
        logger.info("")
        logger.info("3. HARDWARE:")
        logger.info("   - GPU: P100 (recommended)")
        logger.info("   - Training time: ~1-1.5 hours for 50 epochs")
        logger.info("")
        logger.info("4. NEW FEATURES:")
        logger.info("   - Data Augmentation: HorizontalFlip, RandomRotation, ColorJitter, RandomBrightnessContrast, ElasticTransform")
        logger.info("")
        logger.info("✅ Everything is organized and ready!")
        logger.info("[+] Ready for ERROR-FREE Kaggle training!")
    else:
        logger.error("❌ Failed to create zip file")

if __name__ == "__main__":
    main() 