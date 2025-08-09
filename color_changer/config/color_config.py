"""
Hair color configuration module.

Defines available hair colors, their RGB values, tone variations, and default paths for image and model files.
Used throughout the hair color transformation pipeline for consistent color and tone management.
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
    ([240, 220, 170], "Blonde"),
    ([184, 115, 51], "Copper"),
    ([120, 85, 60], "Brown"),
    ([160, 82, 45], "Auburn"),
    ([255, 105, 180], "Pink"),
    ([0, 0, 255], "Blue"),
    ([128, 0, 128], "Purple"),
    ([128, 128, 128], "Gray"),
    ([170, 30, 30], "Red"),
    ([0, 160, 80], "Green"),
    ([0, 170, 160], "Teal")
]

# Color-specific tones - Special variations for each color
CUSTOM_TONES = {
    "Black": {
        "jet": {"saturation_factor": 0.1, "brightness_factor": 0.1, "hue_offset": -2, "description": "Pure jet black"},
        "soft": {"saturation_factor": 0.4, "brightness_factor": 0.7, "hue_offset": +3, "description": "Soft, warm black"},
        "onyx": {"saturation_factor": 0.05, "brightness_factor": 0.05, "hue_offset": -4, "description": "Deep, rich onyx black"},
        "charcoal": {"saturation_factor": 0.3, "brightness_factor": 0.5, "hue_offset": -6, "description": "Charcoal gray-black"}
    },
    "Blonde": {
        "platinum": {"saturation_factor": 0.15, "brightness_factor": 1.55, "intensity": 0.90, "hue_offset": -6, "description": "Ultra-light platinum blonde"},
        "golden": {"saturation_factor": 1.4, "brightness_factor": 1.10, "intensity": 1.00, "hue_offset": +8, "description": "Warm golden blonde"},
        "ash": {"saturation_factor": 0.35, "brightness_factor": 1.25, "intensity": 0.95, "hue_offset": -10, "description": "Cool ash blonde"},
        "honey": {"saturation_factor": 1.30, "brightness_factor": 0.95, "intensity": 1.00, "hue_offset": +6, "description": "Sweet honey blonde"},
        "beige": {"saturation_factor": 0.80, "brightness_factor": 1.05, "intensity": 0.95, "hue_offset": -2, "description": "Neutral beige blonde"}
    },
    "Copper": {
        "bright": {"saturation_factor": 1.70, "brightness_factor": 1.30, "intensity": 1.00, "hue_offset": +4, "description": "Bright copper"},
        "antique": {"saturation_factor": 0.60, "brightness_factor": 0.85, "intensity": 0.90, "hue_offset": -6, "description": "Antique copper"},
        "penny": {"saturation_factor": 1.30, "brightness_factor": 0.90, "intensity": 1.00, "hue_offset": +2, "description": "Penny copper"},
        "rose": {"saturation_factor": 1.50, "brightness_factor": 1.20, "intensity": 1.00, "hue_offset": +8, "description": "Rose copper"}
    },
    "Brown": {
        "chestnut": {"saturation_factor": 1.30, "brightness_factor": 0.95, "intensity": 1.00, "hue_offset": +6, "description": "Rich chestnut brown"},
        "chocolate": {"saturation_factor": 1.20, "brightness_factor": 0.85, "intensity": 0.95, "hue_offset": -4, "description": "Dark chocolate brown"},
        "caramel": {"saturation_factor": 1.40, "brightness_factor": 1.15, "intensity": 1.00, "hue_offset": +10, "description": "Sweet caramel brown"},
        "mahogany": {"saturation_factor": 1.60, "brightness_factor": 0.90, "intensity": 1.00, "hue_offset": +14, "description": "Reddish mahogany brown"},
        "espresso": {"saturation_factor": 1.00, "brightness_factor": 0.80, "intensity": 0.90, "hue_offset": -6, "description": "Deep espresso brown"}
    },
    "Auburn": {
        "classic": {"saturation_factor": 1.20, "brightness_factor": 1.00, "intensity": 1.00, "hue_offset": 0, "description": "Classic auburn"},
        "golden": {"saturation_factor": 1.40, "brightness_factor": 1.15, "intensity": 1.05, "hue_offset": +6, "description": "Golden auburn"},
        "dark": {"saturation_factor": 1.10, "brightness_factor": 0.85, "intensity": 0.95, "hue_offset": -4, "description": "Dark auburn"},
        "copper": {"saturation_factor": 1.60, "brightness_factor": 1.05, "intensity": 1.00, "hue_offset": +10, "description": "Copper auburn"}
    },
    "Pink": {
        "rose": {"saturation_factor": 0.80, "brightness_factor": 1.25, "intensity": 0.95, "hue_offset": +6, "description": "Romantic rose pink"},
        "fuchsia": {"saturation_factor": 1.90, "brightness_factor": 1.00, "intensity": 1.05, "hue_offset": +10, "description": "Vibrant fuchsia pink"},
        "blush": {"saturation_factor": 0.50, "brightness_factor": 1.45, "intensity": 0.90, "hue_offset": -4, "description": "Soft blush pink"},
        "magenta": {"saturation_factor": 1.70, "brightness_factor": 0.90, "intensity": 1.05, "hue_offset": +12, "description": "Bold magenta pink"},
        "coral": {"saturation_factor": 1.30, "brightness_factor": 1.10, "intensity": 1.00, "hue_offset": +8, "description": "Warm coral pink"}
    },
    "Blue": {
        "navy": {"saturation_factor": 2.00, "brightness_factor": 0.85, "intensity": 0.95, "hue_offset": -6, "description": "Deep navy blue"},
        "electric": {"saturation_factor": 2.20, "brightness_factor": 1.25, "intensity": 1.05, "hue_offset": +4, "description": "Electric bright blue"},
        "ice": {"saturation_factor": 0.50, "brightness_factor": 1.50, "intensity": 0.95, "hue_offset": -10, "description": "Ice blue"},
        "midnight": {"saturation_factor": 1.80, "brightness_factor": 0.75, "intensity": 0.90, "hue_offset": -8, "description": "Midnight blue"},
        "sky": {"saturation_factor": 0.80, "brightness_factor": 1.40, "intensity": 1.00, "hue_offset": +6, "description": "Sky blue"}
    },
    "Purple": {
        "violet": {"saturation_factor": 1.80, "brightness_factor": 0.95, "intensity": 1.00, "hue_offset": +6, "description": "Rich violet purple"},
        "lavender": {"saturation_factor": 0.60, "brightness_factor": 1.40, "intensity": 0.95, "hue_offset": -8, "description": "Soft lavender purple"},
        "plum": {"saturation_factor": 1.60, "brightness_factor": 0.80, "intensity": 0.95, "hue_offset": -12, "description": "Deep plum purple"},
        "amethyst": {"saturation_factor": 1.70, "brightness_factor": 1.05, "intensity": 1.00, "hue_offset": +10, "description": "Gemstone amethyst purple"},
        "orchid": {"saturation_factor": 0.90, "brightness_factor": 1.20, "intensity": 0.95, "hue_offset": -4, "description": "Delicate orchid purple"}
    },
    "Gray": {
        "silver": {"saturation_factor": 0.20, "brightness_factor": 1.50, "intensity": 0.95, "hue_offset": +2, "description": "Bright silver gray"},
        "ash": {"saturation_factor": 0.20, "brightness_factor": 1.10, "intensity": 0.95, "hue_offset": -4, "description": "Cool ash gray"},
        "charcoal": {"saturation_factor": 0.15, "brightness_factor": 0.85, "intensity": 0.90, "hue_offset": -6, "description": "Dark charcoal gray"},
        "pearl": {"saturation_factor": 0.25, "brightness_factor": 1.35, "intensity": 1.00, "hue_offset": +4, "description": "Lustrous pearl gray"},
        "steel": {"saturation_factor": 0.20, "brightness_factor": 0.95, "intensity": 0.95, "hue_offset": -2, "description": "Cool steel gray"}
    },
    "Red": {
        "burgundy": {"saturation_factor": 1.50, "brightness_factor": 0.85, "intensity": 0.95, "hue_offset": -8, "description": "Deep burgundy red"},
        "crimson": {"saturation_factor": 1.70, "brightness_factor": 1.05, "intensity": 1.05, "hue_offset": +6, "description": "Vivid crimson red"},
        "rosewood": {"saturation_factor": 1.20, "brightness_factor": 0.95, "intensity": 0.95, "hue_offset": -4, "description": "Muted rosewood red"},
        "scarlet": {"saturation_factor": 1.90, "brightness_factor": 1.15, "intensity": 1.05, "hue_offset": +10, "description": "Bright scarlet red"}
    },
    "Green": {
        "emerald": {"saturation_factor": 1.80, "brightness_factor": 1.05, "intensity": 1.00, "hue_offset": +4, "description": "Rich emerald green"},
        "forest": {"saturation_factor": 1.20, "brightness_factor": 0.90, "intensity": 0.95, "hue_offset": -6, "description": "Deep forest green"},
        "mint": {"saturation_factor": 0.80, "brightness_factor": 1.30, "intensity": 0.95, "hue_offset": +6, "description": "Light mint green"},
        "olive": {"saturation_factor": 0.70, "brightness_factor": 0.95, "intensity": 0.90, "hue_offset": -10, "description": "Subdued olive green"}
    },
    "Teal": {
        "aqua": {"saturation_factor": 1.70, "brightness_factor": 1.10, "intensity": 1.00, "hue_offset": +6, "description": "Bright aqua teal"},
        "deep": {"saturation_factor": 1.30, "brightness_factor": 0.90, "intensity": 0.95, "hue_offset": -4, "description": "Deep ocean teal"},
        "pastel": {"saturation_factor": 0.70, "brightness_factor": 1.30, "intensity": 0.95, "hue_offset": +4, "description": "Soft pastel teal"},
        "seafoam": {"saturation_factor": 1.10, "brightness_factor": 1.15, "intensity": 1.00, "hue_offset": +2, "description": "Fresh seafoam teal"}
    }
}

# Config-driven per-color transformation profiles for unified handler
# These parameters control hue shift strength, saturation scaling/limits,
# value (brightness) gains for shadow/mid/highlight regions, and optional corrections.
# All gains are further scaled by alpha at runtime.

COLOR_PROFILES = {
    "DEFAULT": {
        "hue": {"weight": 0.90},
        "sat": {"scale": 1.2, "min": 0, "max": 255, "approach_target_weight": 0.4, "high_sat_boost": False},
        "val": {"shadow_gain": 1.10, "mid_gain": 1.05, "highlight_gain": 0.95, "bounds": [0, 255]},
        "corrections": {"anti_pink": False, "highlight_tamer": False, "gray_mode": False},
        "lightening": {"enabled": False, "shadow": 0.30, "mid": 0.20, "highlight": 0.08, "desat": 0.08, "dark_thresh": 0.35, "light_thresh": 0.60, "upper_bound": 235}
    },
    "Black": {
        "hue": {"weight": 0.85},
        "sat": {"scale": 0.75, "min": 0, "max": 255, "approach_target_weight": 0.2, "high_sat_boost": False},
        "val": {"shadow_gain": 0.40, "mid_gain": 0.45, "highlight_gain": 0.40, "bounds": [0, 110]},
        "corrections": {"anti_pink": False, "highlight_tamer": True, "gray_mode": False}
    },
    "Blonde": {
        "hue": {"weight": 0.90},
        "sat": {"scale": 1.15, "min": 0, "max": 255, "approach_target_weight": 0.3, "high_sat_boost": False},
        "val": {"shadow_gain": 1.30, "mid_gain": 1.20, "highlight_gain": 0.95, "bounds": [10, 255]},
        "corrections": {"anti_pink": False, "highlight_tamer": True, "gray_mode": False, "highlight_protect": True, "hp_sat_reduce": 0.18, "hp_hue_blend": 0.45, "hp_v_cap": 245, "desat_comp": True, "desat_thresh": 35, "desat_boost": 1.25, "desat_hue_weight_boost": 0.12},
        "lightening": {"enabled": True, "shadow": 0.52, "mid": 0.36, "highlight": 0.16, "desat": 0.08, "dark_thresh": 0.38, "light_thresh": 0.62, "upper_bound": 250}
    },
    "Copper": {
        "hue": {"weight": 0.90},
        "sat": {"scale": 1.60, "min": 150, "max": 255, "approach_target_weight": 0.4, "high_sat_boost": False},
        "val": {"shadow_gain": 1.40, "mid_gain": 1.20, "highlight_gain": 0.90, "bounds": [50, 240]},
        "corrections": {"anti_pink": False, "highlight_tamer": False, "gray_mode": False, "highlight_protect": True, "hp_sat_reduce": 0.15, "hp_hue_blend": 0.35},
        "lightening": {"enabled": True, "shadow": 0.22, "mid": 0.15, "highlight": 0.06, "desat": 0.04, "dark_thresh": 0.40, "light_thresh": 0.55, "upper_bound": 235}
    },
    "Brown": {
        "hue": {"weight": 0.94},
        "sat": {"scale": 1.00, "min": 60, "max": 235, "approach_target_weight": 0.25, "high_sat_boost": False},
        "val": {"shadow_gain": 0.97, "mid_gain": 0.96, "highlight_gain": 0.80, "bounds": [25, 250]},
        "corrections": {"anti_pink": False, "highlight_tamer": True, "gray_mode": False, "highlight_protect": True, "hp_sat_reduce": 0.15, "hp_hue_blend": 0.30},
        "lightening": {"enabled": True, "shadow": 0.60, "mid": 0.42, "highlight": 0.14, "desat": 0.05, "dark_thresh": 0.40, "light_thresh": 0.55, "upper_bound": 250}
    },
    "Auburn": {
        "hue": {"weight": 0.85},
        "sat": {"scale": 1.40, "min": 100, "max": 255, "approach_target_weight": 0.4, "high_sat_boost": False},
        "val": {"shadow_gain": 1.30, "mid_gain": 1.10, "highlight_gain": 0.85, "bounds": [30, 220]},
        "corrections": {"anti_pink": False, "highlight_tamer": False, "gray_mode": False}
    },
    "Pink": {
        "hue": {"weight": 0.90},
        "sat": {"scale": 1.80, "min": 120, "max": 255, "approach_target_weight": 0.5, "high_sat_boost": True},
        "val": {"shadow_gain": 1.40, "mid_gain": 1.15, "highlight_gain": 0.95, "bounds": [40, 230]},
        "corrections": {"anti_pink": False, "highlight_tamer": False, "gray_mode": False},
        "lightening": {"enabled": True, "shadow": 0.45, "mid": 0.30, "highlight": 0.12, "desat": 0.05, "dark_thresh": 0.40, "light_thresh": 0.60, "upper_bound": 245}
    },
    "Blue": {
        "hue": {"weight": 0.90},
        "sat": {"scale": 1.45, "min": 165, "max": 240, "approach_target_weight": 0.40, "high_sat_boost": True},
        "val": {"shadow_gain": 1.18, "mid_gain": 1.05, "highlight_gain": 0.92, "bounds": [25, 240]},
        "corrections": {"anti_pink": False, "highlight_tamer": False, "gray_mode": False, "hue_band": [110, 125], "hue_center": 120, "post_smooth": True},
        "lightening": {"enabled": True, "shadow": 0.35, "mid": 0.25, "highlight": 0.10, "desat": 0.02, "dark_thresh": 0.40, "light_thresh": 0.60, "upper_bound": 240}
    },
    "Purple": {
        "hue": {"weight": 0.92},
        "sat": {"scale": 1.62, "min": 190, "max": 255, "approach_target_weight": 0.50, "high_sat_boost": False},
        "val": {"shadow_gain": 1.15, "mid_gain": 1.00, "highlight_gain": 0.66, "bounds": [60, 205]},
        "corrections": {"anti_pink": True, "highlight_tamer": False, "gray_mode": False, "hue_band": [145, 149], "hue_center": 150, "hue_center_weight": 0.45, "desat_near_pink": True, "post_smooth": True, "value_dependent_hue_center": True, "bilateral_hue_smooth": True},
        "lightening": {"enabled": True, "shadow": 0.32, "mid": 0.22, "highlight": 0.10, "desat": 0.03, "dark_thresh": 0.40, "light_thresh": 0.60, "upper_bound": 238}
    },
    "Gray": {
        "hue": {"weight": 0.60, "suppress_factor": 0.40},
        "sat": {"scale": 0.50, "min": 0, "max": 200, "approach_target_weight": 0.0, "high_sat_boost": False},
        "val": {"shadow_gain": 1.20, "mid_gain": 1.00, "highlight_gain": 0.90, "bounds": [0, 255]},
        "corrections": {"anti_pink": False, "highlight_tamer": False, "gray_mode": True},
        "lightening": {"enabled": True, "shadow": 0.35, "mid": 0.22, "highlight": 0.10, "desat": 0.05, "dark_thresh": 0.40, "light_thresh": 0.65, "upper_bound": 235}
    },
    "Red": {
        "hue": {"weight": 0.88},
        "sat": {"scale": 1.45, "min": 100, "max": 255, "approach_target_weight": 0.5, "high_sat_boost": False},
        "val": {"shadow_gain": 1.22, "mid_gain": 1.05, "highlight_gain": 0.86, "bounds": [20, 240]},
        "corrections": {"anti_pink": False, "highlight_tamer": True, "gray_mode": False}
    },
    "Green": {
        "hue": {"weight": 0.90},
        "sat": {"scale": 1.58, "min": 135, "max": 255, "approach_target_weight": 0.5, "high_sat_boost": True},
        "val": {"shadow_gain": 1.26, "mid_gain": 1.06, "highlight_gain": 0.92, "bounds": [15, 245]},
        "corrections": {"anti_pink": False, "highlight_tamer": False, "gray_mode": False, "hue_band": [70, 90]},
        "lightening": {"enabled": True, "shadow": 0.30, "mid": 0.22, "highlight": 0.10, "desat": 0.03, "dark_thresh": 0.40, "light_thresh": 0.60, "upper_bound": 240}
    },
    "Teal": {
        "hue": {"weight": 0.92},
        "sat": {"scale": 1.60, "min": 150, "max": 255, "approach_target_weight": 0.45, "high_sat_boost": True},
        "val": {"shadow_gain": 1.12, "mid_gain": 1.04, "highlight_gain": 0.92, "bounds": [22, 245]},
        "corrections": {"anti_pink": False, "highlight_tamer": False, "gray_mode": False, "hue_band": [90, 100], "hue_center": 95, "post_smooth": True, "value_dependent_hue_center": True, "bilateral_hue_smooth": True},
        "lightening": {"enabled": True, "shadow": 0.28, "mid": 0.20, "highlight": 0.10, "desat": 0.03, "dark_thresh": 0.40, "light_thresh": 0.60, "upper_bound": 240}
    }
}