"""
Color configuration for hair color changing operations.
"""

from pathlib import Path

# Project root directory (color_changer's parent)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Default paths
PREVIEW_IMAGES_DIR = PROJECT_ROOT / "color_changer" / "test_images"
PREVIEW_RESULTS_DIR = PROJECT_ROOT / "color_changer" / "test_results"
DEFAULT_MODEL_PATH = PROJECT_ROOT / "model" / "trained_models" / "2025-07-14_02-57-22_acc0.9708" / "best_model.pth"

# Predefined colors in RGB format [R, G, B] with names
COLORS = [
    ([0, 0, 0], "Black"),
    ([255, 255, 0], "Yellow"),
    ([220, 20, 60], "Red"),
    ([139, 69, 19], "Brown"),
    ([0, 0, 255], "Blue"),
    ([128, 0, 128], "Purple"),
    ([128, 128, 128], "Gray")
]

# Toning configuration - Dramatic differences for visible results
TONE_TYPES = {
    "light": {
        "name": "Light",
        "saturation_factor": 0.4,  # Much reduced saturation for lighter look
        "brightness_factor": 1.8,  # Much increased brightness
        "description": "Lighter, softer version"
    },
    "natural": {
        "name": "Natural", 
        "saturation_factor": 1.0,  # Original saturation
        "brightness_factor": 1.0,  # Original brightness
        "description": "Original color intensity"
    },
    "vibrant": {
        "name": "Vibrant",
        "saturation_factor": 2.0,  # Much increased saturation
        "brightness_factor": 1.3,  # Increased brightness
        "description": "More intense, vivid version"
    },
    "deep": {
        "name": "Deep",
        "saturation_factor": 1.8,  # Much increased saturation
        "brightness_factor": 0.5,  # Much reduced brightness for deeper look
        "description": "Darker, richer version"
    },
    "muted": {
        "name": "Muted",
        "saturation_factor": 0.2,  # Very reduced saturation
        "brightness_factor": 0.7,  # Reduced brightness
        "description": "Subtle, understated version"
    }
}

# Intensity levels for fine-tuning - More dramatic effects
INTENSITY_LEVELS = {
    "subtle": 0.5,
    "moderate": 0.8, 
    "strong": 1.0,
    "maximum": 1.2
}