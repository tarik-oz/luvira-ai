"""
YOLOv8 Configuration

Configuration settings for YOLOv8 hair segmentation training and inference.
"""

import os
from pathlib import Path

# Project root directory
PROJECT_ROOT = Path(__file__).parent

# Scripts directory
SCRIPTS_DIR = PROJECT_ROOT / "scripts"

# Data paths
DATA_DIR = PROJECT_ROOT / "data"
TRAIN_IMAGES_DIR = DATA_DIR / "images" / "train"
TRAIN_LABELS_DIR = DATA_DIR / "labels" / "train"
VAL_IMAGES_DIR = DATA_DIR / "images" / "val"
VAL_LABELS_DIR = DATA_DIR / "labels" / "val"

# Model paths
MODEL_DIR = PROJECT_ROOT / "model"
BEST_MODEL_PATH = MODEL_DIR / "best.pt"

# Test paths
TEST_IMAGES_DIR = PROJECT_ROOT / "test_images"
OUTPUT_DIR = PROJECT_ROOT / "masks"

# Training configuration
TRAINING_CONFIG = {
    "epochs": 8,
    "batch_size": 2,
    "image_size": 640,
    "confidence": 0.5,
    "iou_threshold": 0.5,
    "save": True,
    "project": "yolov8_hair_segmentation"
}

# Model configuration
MODEL_CONFIG = {
    "nc": 1,  # Number of classes
    "names": ["hair"]  # Class names
}

# Create directories if they don't exist
for directory in [DATA_DIR, TRAIN_IMAGES_DIR, TRAIN_LABELS_DIR, 
                  VAL_IMAGES_DIR, VAL_LABELS_DIR, MODEL_DIR, 
                  TEST_IMAGES_DIR, OUTPUT_DIR, SCRIPTS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

def get_yaml_config():
    """Generate YAML configuration string for YOLOv8 training."""
    yaml_content = f"""path: "{DATA_DIR}"
train: "{TRAIN_IMAGES_DIR.relative_to(DATA_DIR)}"
val: "{VAL_IMAGES_DIR.relative_to(DATA_DIR)}"

nc: {MODEL_CONFIG['nc']}
names: {MODEL_CONFIG['names']}
"""
    return yaml_content

def save_yaml_config(output_path: str = None):
    """Save YAML configuration to file."""
    if output_path is None:
        output_path = PROJECT_ROOT / "config.yaml"
    
    yaml_content = get_yaml_config()
    with open(output_path, 'w') as f:
        f.write(yaml_content)
    print(f"YAML config saved to: {output_path}")

if __name__ == "__main__":
    # Generate and save YAML config
    save_yaml_config() 