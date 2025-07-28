#!/usr/bin/env python3
"""
Kaggle Zip Creator for Hair Segmentation Project - UPDATED VERSION
This script creates a zip file with all necessary files for Kaggle training.
Uses the updated model structure.
"""

import zipfile
import logging
from pathlib import Path
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_kaggle_zip():
    """
    Create a zip file for Kaggle with all necessary model files.
    """
    project_root = Path(__file__).parent.parent.parent
    zip_name = f"hair_segmentation_kaggle_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip"
    zip_path = project_root / zip_name
    
    logger.info(f"Creating Kaggle zip: {zip_name}")
    
    # Files and directories to include
    files_to_include = [
        # Core model files
        "model/__init__.py",
        "model/config.py",
        
        # Data loader files
        "model/data_loader/__init__.py",
        "model/data_loader/base_data_loader.py",
        "model/data_loader/factory_data_loader.py",
        "model/data_loader/lazy_data_loader.py",
        "model/data_loader/lazy_dataset.py",
        "model/data_loader/traditional_data_loader.py",
        "model/data_loader/traditional_dataset.py",
        
        # Model architecture files
        "model/models/__init__.py",
        "model/models/unet_model.py",
        "model/models/attention_unet_model.py",
        
        # Training files
        "model/training/__init__.py",
        "model/training/trainer.py",
        "model/training/callbacks.py",
        "model/training/metrics.py",
        
        # Utility files
        "model/utils/__init__.py",
        "model/utils/augmentation.py",
        "model/utils/data_timestamp.py",
        "model/utils/model_saving.py",
        "model/utils/validators.py",
        
        # Kaggle-specific files
        "model/kaggle/__init__.py",
        "model/kaggle/config.py",
        "model/kaggle/train.py",
        "model/kaggle/setup.py",
        
        # Root requirements
        "requirements.txt",
    ]
    
    try:
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for file_path in files_to_include:
                full_path = project_root / file_path
                if full_path.exists():
                    zipf.write(full_path, file_path)
                    logger.info(f"[+] Added: {file_path}")
                else:
                    logger.warning(f"[-] Missing: {file_path}")
            
            # Add main training script for Kaggle
            main_script = '''#!/usr/bin/env python3
"""
Main Kaggle training script for hair segmentation.
"""

import sys
from pathlib import Path

# Add current directory to path
sys.path.insert(0, "/kaggle/working")

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
            zipf.writestr("main_kaggle_train.py", main_script)
            logger.info("[+] Added: main_kaggle_train.py")
            
            # Add Kaggle README
            readme = '''# Hair Segmentation - Kaggle Training

## Quick Start

1. Upload this zip as a Kaggle dataset
2. In your notebook, extract it:
   ```python
   import zipfile
   with zipfile.ZipFile('/kaggle/input/your-dataset/hair_segmentation_kaggle_*.zip', 'r') as zipf:
       zipf.extractall('/kaggle/working/')
   ```
3. Run training:
   ```python
   %cd /kaggle/working
   !python main_kaggle_train.py
   ```

## Configuration

Edit `model/kaggle/config.py` for:
- Dataset name in Kaggle
- Batch size and epochs
- Model architecture choice
'''
            zipf.writestr("README_KAGGLE.md", readme)
            logger.info("[+] Added: README_KAGGLE.md")
    
        logger.info(f"\n✅ Zip created: {zip_path}")
        return zip_path
        
    except Exception as e:
        logger.error(f"❌ Error: {e}")
        return None

if __name__ == "__main__":
    create_kaggle_zip()
