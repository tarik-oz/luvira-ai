"""
Configuration file for Hair Segmentation API.
Contains all API-related configurable parameters.
"""

import os
from pathlib import Path

# Project root directory (going up from api/ to project root)
PROJECT_ROOT = Path(__file__).parent.parent

# API configuration
API_CONFIG = {
    "default_model_path": str(PROJECT_ROOT / "model" / "models" / "trained_models" / "2025-07-06_23-00-07_acc0.8727" / "best_model.pth"),
    "host": "0.0.0.0",
    "port": 8000,
    "reload": True,
    "log_level": "info",
    "cors_origins": ["*"],
    "cors_methods": ["*"],
    "cors_headers": ["*"],
    "max_file_size": 10 * 1024 * 1024,  # 10MB
    "allowed_image_types": ["image/jpeg", "image/png", "image/jpg"]
} 