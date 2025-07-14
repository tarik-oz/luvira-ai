#!/usr/bin/env python3
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
        print("
🎯 Ready to start training!")
        print("Run: !python /kaggle/working/main_kaggle_train.py")
        
        return True
        
    except Exception as e:
        print(f"❌ Error extracting files: {e}")
        return False

if __name__ == "__main__":
    success = extract_kaggle_files()
    if success:
        print("
" + "="*50)
        print("🚀 READY FOR TRAINING!")
        print("="*50)
    else:
        print("
" + "="*50) 
        print("❌ SETUP FAILED - Please check the errors above")
        print("="*50)
