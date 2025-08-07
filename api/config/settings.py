"""
Configuration settings for Hair Segmentation API
"""

from pathlib import Path

# Project root directory (going up from api/ to project root)
PROJECT_ROOT = Path(__file__).parent.parent.parent

# API configuration
API_CONFIG = {
    "host": "0.0.0.0",
    "port": 8000,
    "reload": True,
    "log_level": "info",
    "cors_origins": ["*"],
    "cors_methods": ["*"],
    "cors_headers": ["*"],
    "api_title": "Hair Segmentation API",
    "api_description": "API for hair segmentation using deep learning model",
    "api_version": "1.1.0"
}

# File validation settings
FILE_VALIDATION = {
    "max_file_size": 10 * 1024 * 1024,  # 10MB
    "allowed_image_types": ["image/jpeg", "image/png", "image/jpg"],
    "max_filename_length": 255,
    "min_image_dimensions": (64, 64),
    "max_image_dimensions": (4096, 4096)
}

# Model settings
MODEL_CONFIG = {
    "default_model_path": str(PROJECT_ROOT / "model" / "trained_models" / "2025-07-15_06-54-35_acc0.9757" / "best_model.pth"),
    "prediction_timeout": 30,  # seconds
    "color_change_timeout": 30,  # seconds
    "device_preference": "auto"  # auto, cpu or cuda
}

# Session/Cache settings
SESSION_CONFIG = {
    "cache_dir": PROJECT_ROOT / "session_data",
    "session_timeout_minutes": 30,
    "cleanup_interval_minutes": 10,
    "auto_cleanup_on_startup": True,
    "auto_cleanup_on_shutdown": True
} 