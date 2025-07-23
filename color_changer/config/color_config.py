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
    ([250, 240, 190], "Blonde"),
    ([184, 115, 51], "Copper"),
    ([139, 69, 19], "Brown"),
    ([160, 82, 45], "Auburn"),
    ([255, 105, 180], "Pink"),
    ([0, 0, 255], "Blue"),
    ([128, 0, 128], "Purple"),
    ([128, 128, 128], "Gray")
]

# Color-specific tones - Special variations for each color
CUSTOM_TONES = {
    "Black": {
        "jet": {"saturation_factor": 0.1, "brightness_factor": 0.1, "description": "Pure jet black"},
        "soft": {"saturation_factor": 0.4, "brightness_factor": 0.7, "description": "Soft, warm black"},
        "onyx": {"saturation_factor": 0.05, "brightness_factor": 0.05, "description": "Deep, rich onyx black"},
        "charcoal": {"saturation_factor": 0.3, "brightness_factor": 0.5, "description": "Charcoal gray-black"}
    },
    "Blonde": {
        "platinum": {"saturation_factor": 0.1, "brightness_factor": 1.6, "description": "Ultra-light platinum blonde"},
        "golden": {"saturation_factor": 1.8, "brightness_factor": 1.1, "description": "Warm golden blonde"},
        "ash": {"saturation_factor": 0.2, "brightness_factor": 1.3, "description": "Cool ash blonde"},
        "honey": {"saturation_factor": 1.4, "brightness_factor": 0.9, "description": "Sweet honey blonde"},
        "strawberry": {"saturation_factor": 1.6, "brightness_factor": 0.8, "description": "Strawberry blonde with red hints"}
    },
    "Copper": {
        "bright": {"saturation_factor": 2.0, "brightness_factor": 1.5, "description": "Bright copper"},
        "antique": {"saturation_factor": 0.4, "brightness_factor": 0.4, "description": "Antique copper"},
        "penny": {"saturation_factor": 1.5, "brightness_factor": 0.7, "description": "Penny copper"},
        "rose": {"saturation_factor": 1.8, "brightness_factor": 1.3, "description": "Rose copper"}
    },
    "Brown": {
        "chestnut": {"saturation_factor": 1.6, "brightness_factor": 0.7, "description": "Rich chestnut brown"},
        "chocolate": {"saturation_factor": 1.3, "brightness_factor": 0.4, "description": "Dark chocolate brown"},
        "caramel": {"saturation_factor": 1.8, "brightness_factor": 1.2, "description": "Sweet caramel brown"},
        "mahogany": {"saturation_factor": 2.0, "brightness_factor": 0.6, "description": "Reddish mahogany brown"},
        "espresso": {"saturation_factor": 0.8, "brightness_factor": 0.2, "description": "Deep espresso brown"}
    },
    "Auburn": {
        "classic": {"saturation_factor": 1.2, "brightness_factor": 0.8, "description": "Classic auburn"},
        "golden": {"saturation_factor": 2.0, "brightness_factor": 1.3, "description": "Golden auburn"},
        "dark": {"saturation_factor": 1.5, "brightness_factor": 0.3, "description": "Dark auburn"},
        "copper": {"saturation_factor": 2.2, "brightness_factor": 1.0, "description": "Copper auburn"}
    },
    "Pink": {
        "rose": {"saturation_factor": 0.6, "brightness_factor": 1.3, "description": "Romantic rose pink"},
        "fuchsia": {"saturation_factor": 2.0, "brightness_factor": 0.9, "description": "Vibrant fuchsia pink"},
        "blush": {"saturation_factor": 0.3, "brightness_factor": 1.5, "description": "Soft blush pink"},
        "magenta": {"saturation_factor": 1.8, "brightness_factor": 0.7, "description": "Bold magenta pink"},
        "coral": {"saturation_factor": 1.4, "brightness_factor": 1.1, "description": "Warm coral pink"}
    },
    "Blue": {
        "navy": {"saturation_factor": 2.2, "brightness_factor": 0.2, "description": "Deep navy blue"},
        "electric": {"saturation_factor": 2.5, "brightness_factor": 1.8, "description": "Electric bright blue"},
        "ice": {"saturation_factor": 0.2, "brightness_factor": 1.9, "description": "Ice blue"},
        "midnight": {"saturation_factor": 2.0, "brightness_factor": 0.1, "description": "Midnight blue"},
        "sky": {"saturation_factor": 0.4, "brightness_factor": 1.7, "description": "Sky blue"}
    },
    "Purple": {
        "violet": {"saturation_factor": 2.0, "brightness_factor": 0.6, "description": "Rich violet purple"},
        "lavender": {"saturation_factor": 0.3, "brightness_factor": 1.7, "description": "Soft lavender purple"},
        "plum": {"saturation_factor": 1.8, "brightness_factor": 0.3, "description": "Deep plum purple"},
        "amethyst": {"saturation_factor": 1.9, "brightness_factor": 1.1, "description": "Gemstone amethyst purple"},
        "orchid": {"saturation_factor": 0.6, "brightness_factor": 1.4, "description": "Delicate orchid purple"}
    },
    "Gray": {
        "silver": {"saturation_factor": 0.05, "brightness_factor": 1.8, "description": "Bright silver gray"},
        "ash": {"saturation_factor": 0.1, "brightness_factor": 0.8, "description": "Cool ash gray"},
        "charcoal": {"saturation_factor": 0.08, "brightness_factor": 0.3, "description": "Dark charcoal gray"},
        "pearl": {"saturation_factor": 0.15, "brightness_factor": 1.5, "description": "Lustrous pearl gray"},
        "steel": {"saturation_factor": 0.12, "brightness_factor": 0.6, "description": "Cool steel gray"}
    }
}