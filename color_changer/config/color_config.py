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
    ([15, 15, 15], "Black"),
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
        # Use absolute target RGBs for robust tone separation on difficult bases
        "jet": {"rgb": [2, 2, 2], "saturation_factor": 0.8, "brightness_factor": 0.6, "hue_offset": -2, "description": "Pure jet black"},
        "soft": {"rgb": [12, 11, 11], "saturation_factor": 0.9, "brightness_factor": 0.9, "hue_offset": +3, "description": "Soft, warm black"},
        "onyx": {"rgb": [1, 1, 1], "saturation_factor": 0.8, "brightness_factor": 0.5, "hue_offset": -4, "description": "Deep, rich onyx black"},
        "charcoal": {"rgb": [8, 8, 9], "saturation_factor": 0.9, "brightness_factor": 0.8, "hue_offset": -6, "description": "Charcoal gray-black"}
    },
    "Blonde": {
        "platinum": {"rgb": [255, 252, 245], "saturation_factor": 0.8, "brightness_factor": 1.5, "intensity": 0.90, "hue_offset": -6, "description": "Ultra-light platinum blonde"},
        "golden": {"rgb": [255, 226, 142], "saturation_factor": 1.30, "brightness_factor": 1.05, "intensity": 1.00, "hue_offset": +8, "description": "Warm golden blonde"},
        "ash": {"rgb": [223, 218, 201], "saturation_factor": 0.6, "brightness_factor": 1.15, "intensity": 0.95, "hue_offset": -10, "description": "Cool ash blonde"},
        "honey": {"rgb": [232, 201, 140], "saturation_factor": 1.20, "brightness_factor": 1.00, "intensity": 1.00, "hue_offset": +6, "description": "Sweet honey blonde"},
        "beige": {"rgb": [255, 230, 185], "saturation_factor": 0.9, "brightness_factor": 1.10, "intensity": 0.95, "hue_offset": -2, "description": "Neutral beige blonde"}
    },
    "Copper": {
        "bright": {"rgb": [255, 127, 28], "saturation_factor": 1.4, "brightness_factor": 1.20, "intensity": 1.00, "hue_offset": +4, "description": "Bright copper"},
        "antique": {"rgb": [134, 91, 58], "saturation_factor": 0.8, "brightness_factor": 0.95, "intensity": 0.90, "hue_offset": -6, "description": "Antique copper"},
        "penny": {"rgb": [188, 102, 38], "saturation_factor": 1.2, "brightness_factor": 0.95, "intensity": 1.00, "hue_offset": +2, "description": "Penny copper"},
        "rose": {"rgb": [255, 122, 53], "saturation_factor": 1.35, "brightness_factor": 1.10, "intensity": 1.00, "hue_offset": +8, "description": "Rose copper"}
    },
    "Brown": {
        "chestnut": {"rgb": [138, 87, 46], "saturation_factor": 1.10, "brightness_factor": 0.95, "intensity": 1.00, "hue_offset": +6, "description": "Rich chestnut brown"},
        "chocolate": {"rgb": [101, 74, 58], "saturation_factor": 1.00, "brightness_factor": 0.85, "intensity": 0.95, "hue_offset": -4, "description": "Dark chocolate brown"},
        "caramel": {"rgb": [159, 107, 54], "saturation_factor": 1.20, "brightness_factor": 1.10, "intensity": 1.00, "hue_offset": +10, "description": "Sweet caramel brown"},
        "mahogany": {"rgb": [142, 73, 40], "saturation_factor": 1.30, "brightness_factor": 0.92, "intensity": 1.00, "hue_offset": +14, "description": "Reddish mahogany brown"},
        "espresso": {"rgb": [91, 68, 52], "saturation_factor": 0.95, "brightness_factor": 0.85, "intensity": 0.90, "hue_offset": -6, "description": "Deep espresso brown"}
    },
    "Auburn": {
        "classic": {"rgb": [174, 78, 30], "saturation_factor": 1.10, "brightness_factor": 1.00, "intensity": 1.00, "hue_offset": 0, "description": "Classic auburn"},
        "golden": {"rgb": [212, 98, 25], "saturation_factor": 1.25, "brightness_factor": 1.10, "intensity": 1.05, "hue_offset": +6, "description": "Golden auburn"},
        "dark": {"rgb": [131, 71, 41], "saturation_factor": 1.00, "brightness_factor": 0.90, "intensity": 0.95, "hue_offset": -4, "description": "Dark auburn"},
        "rust": {"rgb": [179, 76, 22], "saturation_factor": 1.30, "brightness_factor": 0.95, "intensity": 1.00, "hue_offset": +4, "description": "Rust auburn"}
    },
    "Pink": {
        "rose": {"rgb": [255, 137, 182], "saturation_factor": 1.00, "brightness_factor": 1.05, "intensity": 0.95, "hue_offset": +6, "description": "Romantic rose pink"},
        "fuchsia": {"rgb": [255, 59, 189], "saturation_factor": 1.40, "brightness_factor": 1.00, "intensity": 1.05, "hue_offset": +10, "description": "Vibrant fuchsia pink"},
        "blush": {"rgb": [255, 191, 219], "saturation_factor": 0.80, "brightness_factor": 1.20, "intensity": 0.90, "hue_offset": -4, "description": "Soft blush pink"},
        "magenta": {"rgb": [230, 42, 178], "saturation_factor": 1.50, "brightness_factor": 0.95, "intensity": 1.05, "hue_offset": +12, "description": "Bold magenta pink"},
        "coral": {"rgb": [255, 110, 153], "saturation_factor": 1.20, "brightness_factor": 1.05, "intensity": 1.00, "hue_offset": +8, "description": "Warm coral pink"}
    },
    "Blue": {
        "navy": {"rgb": [0, 0, 217], "saturation_factor": 1.60, "brightness_factor": 0.90, "intensity": 0.95, "hue_offset": -6, "description": "Deep navy blue"},
        "electric": {"rgb": [0, 68, 255], "saturation_factor": 1.80, "brightness_factor": 1.15, "intensity": 1.05, "hue_offset": +4, "description": "Electric bright blue"},
        "ice": {"rgb": [168, 168, 255], "saturation_factor": 0.60, "brightness_factor": 1.35, "intensity": 0.95, "hue_offset": -10, "description": "Ice blue"},
        "midnight": {"rgb": [0, 19, 191], "saturation_factor": 1.40, "brightness_factor": 0.85, "intensity": 0.90, "hue_offset": -8, "description": "Midnight blue"},
        "sky": {"rgb": [89, 89, 255], "saturation_factor": 0.90, "brightness_factor": 1.25, "intensity": 1.00, "hue_offset": +6, "description": "Sky blue"}
    },
    "Purple": {
        "violet": {"rgb": [142, 0, 142], "saturation_factor": 1.50, "brightness_factor": 1.00, "intensity": 1.00, "hue_offset": +6, "description": "Rich violet purple"},
        "lavender": {"rgb": [161, 106, 179], "saturation_factor": 0.80, "brightness_factor": 1.25, "intensity": 0.95, "hue_offset": -8, "description": "Soft lavender purple"},
        "plum": {"rgb": [91, 0, 118], "saturation_factor": 1.30, "brightness_factor": 0.85, "intensity": 0.95, "hue_offset": -12, "description": "Deep plum purple"},
        "amethyst": {"rgb": [161, 0, 161], "saturation_factor": 1.55, "brightness_factor": 1.05, "intensity": 1.00, "hue_offset": +10, "description": "Gemstone amethyst purple"},
        "orchid": {"rgb": [151, 61, 153], "saturation_factor": 0.95, "brightness_factor": 1.15, "intensity": 0.95, "hue_offset": -4, "description": "Delicate orchid purple"}
    },
    "Gray": {
        "silver": {"rgb": [194, 190, 189], "saturation_factor": 0.25, "brightness_factor": 1.35, "intensity": 0.95, "hue_offset": +2, "description": "Bright silver gray"},
        "ash": {"rgb": [140, 142, 142], "saturation_factor": 0.25, "brightness_factor": 1.10, "intensity": 0.95, "hue_offset": -4, "description": "Cool ash gray"},
        "charcoal": {"rgb": [108, 109, 109], "saturation_factor": 0.18, "brightness_factor": 0.90, "intensity": 0.90, "hue_offset": -6, "description": "Dark charcoal gray"},
        "pearl": {"rgb": [176, 171, 170], "saturation_factor": 0.30, "brightness_factor": 1.25, "intensity": 1.00, "hue_offset": +4, "description": "Lustrous pearl gray"},
        "steel": {"rgb": [121, 122, 122], "saturation_factor": 0.22, "brightness_factor": 1.00, "intensity": 0.95, "hue_offset": -2, "description": "Cool steel gray"}
    },
    "Red": {
        "burgundy": {"rgb": [134, 25, 41], "saturation_factor": 1.30, "brightness_factor": 0.90, "intensity": 0.95, "hue_offset": -8, "description": "Deep burgundy red"},
        "cherry": {"rgb": [139, 0, 38], "saturation_factor": 1.35, "brightness_factor": 0.90, "intensity": 0.95, "hue_offset": -8, "description": "Deep cherry red"},
        "rose": {"rgb": [222, 49, 99], "saturation_factor": 1.50, "brightness_factor": 1.15, "intensity": 1.10, "hue_offset": +15, "description": "Soft rose red"},
        "crimson": {"rgb": [209, 29, 21], "saturation_factor": 1.55, "brightness_factor": 1.05, "intensity": 1.05, "hue_offset": +6, "description": "Vivid crimson red"},
        "scarlet": {"rgb": [236, 42, 29], "saturation_factor": 1.70, "brightness_factor": 1.10, "intensity": 1.05, "hue_offset": +10, "description": "Bright scarlet red"},
    },
    "Green": {
        "emerald": {"rgb": [0, 179, 81], "saturation_factor": 1.50, "brightness_factor": 1.05, "intensity": 1.00, "hue_offset": +4, "description": "Rich emerald green"},
        "forest": {"rgb": [0, 131, 80], "saturation_factor": 1.10, "brightness_factor": 0.95, "intensity": 0.95, "hue_offset": -6, "description": "Deep forest green"},
        "mint": {"rgb": [78, 223, 140], "saturation_factor": 1.10, "brightness_factor": 1.25, "intensity": 0.95, "hue_offset": +6, "description": "Light mint green"},
        "olive": {"rgb": [53, 131, 88], "saturation_factor": 0.95, "brightness_factor": 1.00, "intensity": 0.90, "hue_offset": -10, "description": "Subdued olive green"}
    },
    "Teal": {
        "aqua": {"rgb": [0, 201, 170], "saturation_factor": 1.35, "brightness_factor": 1.10, "intensity": 1.00, "hue_offset": +6, "description": "Bright aqua teal"},
        "deep": {"rgb": [0, 141, 140], "saturation_factor": 1.15, "brightness_factor": 0.95, "intensity": 0.95, "hue_offset": -4, "description": "Deep ocean teal"},
        "pastel": {"rgb": [84, 235, 214], "saturation_factor": 0.85, "brightness_factor": 1.25, "intensity": 0.95, "hue_offset": +4, "description": "Soft pastel teal"},
        "seafoam": {"rgb": [0, 210, 189], "saturation_factor": 1.20, "brightness_factor": 1.10, "intensity": 1.00, "hue_offset": +2, "description": "Fresh seafoam teal"}
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
        "hue": {"weight": 0.65},
        "sat": {"scale": 0.65, "min": 0, "max": 255, "approach_target_weight": 0.20, "high_sat_boost": False},
        "val": {"shadow_gain": 0.40, "mid_gain": 0.45, "highlight_gain": 0.50, "bounds": [0, 125]},
        "corrections": {"anti_pink": False, "highlight_tamer": True, "gray_mode": False}
    },
    "Blonde": {
        "hue": {"weight": 0.90},
        "sat": {"scale": 1.10, "min": 0, "max": 255, "approach_target_weight": 0.28, "high_sat_boost": False},
        "val": {"shadow_gain": 1.62, "mid_gain": 1.34, "highlight_gain": 1.16, "bounds": [10, 252]},
        "corrections": {"anti_pink": False, "highlight_tamer": True, "gray_mode": False, "highlight_protect": True, "hp_sat_reduce": 0.20, "hp_hue_blend": 0.48, "hp_v_cap": 242, "desat_comp": True, "desat_thresh": 35, "desat_boost": 1.20, "desat_hue_weight_boost": 0.10, "post_smooth": True, "bilateral_hue_smooth": True},
        "lightening": {"enabled": True, "shadow": 0.52, "mid": 0.36, "highlight": 0.16, "desat": 0.08, "dark_thresh": 0.38, "light_thresh": 0.62, "upper_bound": 250}
    },
    "Copper": {
        "hue": {"weight": 0.85},
        "sat": {"scale": 1.28, "min": 40, "max": 220, "approach_target_weight": 0.26, "high_sat_boost": False},
        "val": {"shadow_gain": 1.56, "mid_gain": 1.30, "highlight_gain": 0.92, "bounds": [50, 236]},
        "corrections": {"anti_pink": False, "highlight_tamer": False, "gray_mode": False, "hue_band": [12, 14], "hue_center": 13, "hue_center_weight": 0.20, "highlight_protect": True, "hp_sat_reduce": 0.42, "hp_hue_blend": 0.24, "hp_v_cap": 238, "post_smooth": True, "bilateral_hue_smooth": True, "value_dependent_hue_center": True},
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
        "corrections": {"anti_pink": False, "highlight_tamer": False, "gray_mode": False, "hue_band": [11, 14]}
    },
    "Pink": {
        "hue": {"weight": 0.90},
        "sat": {"scale": 1.70, "min": 120, "max": 255, "approach_target_weight": 0.45, "high_sat_boost": True},
        "val": {"shadow_gain": 1.40, "mid_gain": 1.15, "highlight_gain": 0.95, "bounds": [40, 230]},
        "corrections": {"anti_pink": False, "highlight_tamer": False, "gray_mode": False, "hue_band": [165, 175], "post_smooth": True, "bilateral_hue_smooth": True},
        "lightening": {"enabled": True, "shadow": 0.45, "mid": 0.30, "highlight": 0.12, "desat": 0.05, "dark_thresh": 0.40, "light_thresh": 0.60, "upper_bound": 245}
    },
    "Blue": {
        "hue": {"weight": 0.90},
        "sat": {"scale": 1.45, "min": 165, "max": 240, "approach_target_weight": 0.40, "high_sat_boost": True},
        "val": {"shadow_gain": 1.18, "mid_gain": 1.05, "highlight_gain": 0.92, "bounds": [25, 240]},
        "corrections": {"anti_pink": False, "highlight_tamer": False, "gray_mode": False, "hue_band": [114, 120], "hue_center": 120, "post_smooth": True, "bilateral_hue_smooth": True},
        "lightening": {"enabled": True, "shadow": 0.35, "mid": 0.25, "highlight": 0.10, "desat": 0.02, "dark_thresh": 0.40, "light_thresh": 0.60, "upper_bound": 240}
    },
    "Purple": {
        "hue": {"weight": 0.92},
        "sat": {"scale": 1.55, "min": 150, "max": 255, "approach_target_weight": 0.38, "high_sat_boost": False},
        "val": {"shadow_gain": 1.6, "mid_gain": 1.48, "highlight_gain": 1.06, "bounds": [60, 205]},
        "corrections": {"anti_pink": True, "highlight_tamer": False, "gray_mode": False, "hue_band": [145, 149], "hue_center": 150, "hue_center_weight": 0.45, "desat_near_pink": True, "post_smooth": True, "value_dependent_hue_center": True, "bilateral_hue_smooth": True},
        "lightening": {"enabled": True, "shadow": 0.4, "mid": 0.28, "highlight": 0.10, "desat": 0.03, "dark_thresh": 0.40, "light_thresh": 0.50, "upper_bound": 245}
    },
    "Gray": {
        "hue": {"weight": 0.60, "suppress_factor": 0.40},
        "sat": {"scale": 0.48, "min": 0, "max": 200, "approach_target_weight": 0.0, "high_sat_boost": False},
        "val": {"shadow_gain": 1.20, "mid_gain": 1.00, "highlight_gain": 0.90, "bounds": [0, 255]},
        "corrections": {"anti_pink": False, "highlight_tamer": False, "gray_mode": True, "post_smooth": True},
        "lightening": {"enabled": True, "shadow": 0.35, "mid": 0.22, "highlight": 0.10, "desat": 0.05, "dark_thresh": 0.40, "light_thresh": 0.65, "upper_bound": 235}
    },
    "Red": {
        "hue": {"weight": 0.88},
        "sat": {"scale": 1.40, "min": 100, "max": 255, "approach_target_weight": 0.48, "high_sat_boost": False},
        "val": {"shadow_gain": 1.20, "mid_gain": 1.03, "highlight_gain": 0.84, "bounds": [30, 228]},
        "corrections": {"anti_pink": False, "highlight_tamer": True, "gray_mode": False, "hue_band": [4, 8], "post_smooth": True}
    },
    "Green": {
        "hue": {"weight": 0.87},
        "sat": {"scale": 1.58, "min": 135, "max": 255, "approach_target_weight": 0.5, "high_sat_boost": True},
        "val": {"shadow_gain": 1.26, "mid_gain": 1.06, "highlight_gain": 0.92, "bounds": [25, 235]},
        "corrections": {"anti_pink": False, "highlight_tamer": False, "gray_mode": False, "hue_band": [78, 80]},
        "lightening": {"enabled": True, "shadow": 0.30, "mid": 0.22, "highlight": 0.10, "desat": 0.03, "dark_thresh": 0.40, "light_thresh": 0.60, "upper_bound": 240}
    },
    "Teal": {
        "hue": {"weight": 0.92},
        "sat": {"scale": 1.60, "min": 150, "max": 255, "approach_target_weight": 0.45, "high_sat_boost": True},
        "val": {"shadow_gain": 1.12, "mid_gain": 1.04, "highlight_gain": 0.92, "bounds": [22, 245]},
        "corrections": {"anti_pink": False, "highlight_tamer": False, "gray_mode": False, "hue_band": [93, 97], "hue_center": 95, "post_smooth": True, "value_dependent_hue_center": True, "bilateral_hue_smooth": True},
        "lightening": {"enabled": True, "shadow": 0.28, "mid": 0.20, "highlight": 0.10, "desat": 0.03, "dark_thresh": 0.40, "light_thresh": 0.60, "upper_bound": 240}
    }
}