"""
Configuration settings for Hair Segmentation API (env-driven)
"""

import os
from pathlib import Path
from typing import List

# Project root directory (going up from api/ to project root)
PROJECT_ROOT = Path(__file__).parent.parent.parent


def _get_bool(name: str, default: bool) -> bool:
    val = os.getenv(name)
    if val is None:
        return default
    return str(val).strip().lower() in {"1", "true", "yes", "on"}


def _get_int(name: str, default: int) -> int:
    val = os.getenv(name)
    try:
        return int(val) if val is not None else default
    except Exception:
        return default


def _get_float(name: str, default: float) -> float:
    val = os.getenv(name)
    try:
        return float(val) if val is not None else default
    except Exception:
        return default


def _get_list(name: str, default: List[str]) -> List[str]:
    raw = os.getenv(name)
    if not raw:
        return default
    return [s.strip() for s in raw.split(',') if s.strip()]


# API configuration
API_CONFIG = {
    "host": os.getenv("HOST", "0.0.0.0"),
    "port": _get_int("PORT", 8000),
    "reload": _get_bool("RELOAD", True),
    "log_level": os.getenv("LOG_LEVEL", "info"),
    "cors_origins": _get_list("CORS_ORIGINS", ["*"]),
    "cors_methods": _get_list("CORS_METHODS", ["*"]),
    "cors_headers": _get_list("CORS_HEADERS", ["*"]),
    "api_title": "Hair Segmentation API",
    "api_description": "API for hair segmentation using deep learning model",
    "api_version": "1.1.0"
}

# File validation settings
FILE_VALIDATION = {
    "max_file_size": _get_int("MAX_FILE_SIZE", 10 * 1024 * 1024),  # 10MB
    "allowed_image_types": _get_list("ALLOWED_IMAGE_TYPES", ["image/jpeg", "image/png", "image/jpg"]),
    "max_filename_length": _get_int("MAX_FILENAME_LENGTH", 255),
    "min_image_dimensions": (
        _get_int("MIN_IMAGE_WIDTH", 64),
        _get_int("MIN_IMAGE_HEIGHT", 64),
    ),
    "max_image_dimensions": (
        _get_int("MAX_IMAGE_WIDTH", 4096),
        _get_int("MAX_IMAGE_HEIGHT", 4096),
    ),
}

# Model settings
_default_model_path = os.getenv(
    "MODEL_LOCAL_PATH",
    str(PROJECT_ROOT / "model" / "trained_models" / "2025-07-15_06-54-35_acc0.9757" / "best_model.pth"),
)

MODEL_CONFIG = {
    "default_model_path": _default_model_path,
    "model_s3_uri": os.getenv("MODEL_S3_URI", ""),  # optional: s3://bucket/path/to/model.pth
    "prediction_timeout": _get_int("PREDICTION_TIMEOUT", 30),  # seconds
    "color_change_timeout": _get_int("COLOR_CHANGE_TIMEOUT", 30),  # seconds
    "device_preference": os.getenv("DEVICE_PREFERENCE", "auto"),  # auto, cpu or cuda
    # Hair presence validation (upload-only gate)
    # Pixels above this grayscale value are counted as hair for presence check
    "hair_presence_pixel_threshold": _get_int("HAIR_PRESENCE_PIXEL_THRESHOLD", 35),
    # Minimal ratio of hair pixels to accept image (e.g., 0.5%)
    "minimal_hair_ratio": _get_float("MINIMAL_HAIR_RATIO", 0.005),
}

# Session/Cache settings
SESSION_CONFIG = {
    # backend: 'filesystem' (local dev) or 's3' (production)
    "backend": os.getenv("SESSION_BACKEND", "filesystem").lower(),
    "cache_dir": Path(os.getenv("SESSION_CACHE_DIR", str(PROJECT_ROOT / "session_data"))),
    "session_timeout_minutes": _get_int("SESSION_TIMEOUT_MINUTES", 30),
    "cleanup_interval_minutes": _get_int("SESSION_CLEANUP_INTERVAL_MINUTES", 10),
    "auto_cleanup_on_startup": _get_bool("AUTO_CLEANUP_ON_STARTUP", True),
    "auto_cleanup_on_shutdown": _get_bool("AUTO_CLEANUP_ON_SHUTDOWN", True),
    # S3 specific configs
    "s3_bucket": os.getenv("S3_BUCKET", ""),
    "s3_prefix": os.getenv("S3_PREFIX", "sessions/"),
    "s3_region": os.getenv("S3_REGION", ""),
}